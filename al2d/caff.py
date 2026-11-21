"""Read and write CAFF archives (the container of a Cubism .cmo3 project).

Layout (all integers big-endian), as described by StretchyStudio's MIT-licensed caffPacker.js and
moc3ingbird's cmo3.hexpat:

  "CAFF", 3 bytes archive version, 4 bytes format id, 3 bytes format version,
  int32 obfuscation key (plain), 8 reserved bytes,
  preview block (28 bytes),
  int32 entry count,
  per entry: string path, string tag, int64 start, int32 size, bool obfuscated, byte compression, 8 reserved,
  entry data, then two guard bytes 98 99.

Everything after the preview block is XORed with the key (integers with the whole key, bytes with
its low byte).  A compressed entry (compression 33) is a partial zip stream holding one member,
"contents": local file header, raw DEFLATE data, data descriptor, and no central directory.

Command line:
  python caff.py list   <file.cmo3>
  python caff.py unpack <file.cmo3> <folder>
  python caff.py repack <original.cmo3> <new main.xml> <out.cmo3>   (swap main.xml, keep the rest)
"""
import os
import struct
import sys
import time
import zlib

COMPRESS_RAW = 16
COMPRESS_FAST = 33


class Entry:
    def __init__(self, path, tag, data, obfuscated=True, compression=COMPRESS_RAW):
        self.path = path
        self.tag = tag
        self.data = data            # decoded content (unzipped when compressed)
        self.obfuscated = obfuscated
        self.compression = compression


class Archive:
    def __init__(self):
        self.header = None          # the 4+3+4+3 identification bytes
        self.key = 42
        self.reserved = bytes(8)
        self.preview = None         # raw 28-byte preview block
        self.entries = []

    def get(self, path):
        for entry in self.entries:
            if entry.path == path:
                return entry
        raise KeyError(path)


def _mask64(key):
    """The 64-bit XOR mask the format derives from the signed 32-bit key."""
    low = key & 0xFFFFFFFF
    high = 0xFFFFFFFF if key < 0 else low
    return (high << 32) | low


class _Reader:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def take(self, count):
        chunk = self.data[self.pos:self.pos + count]
        self.pos += count
        return chunk

    def byte(self, key=0):
        return self.take(1)[0] ^ (key & 0xFF)

    def int32(self, key=0):
        return (struct.unpack('>I', self.take(4))[0] ^ (key & 0xFFFFFFFF))

    def int64(self, key=0):
        return struct.unpack('>Q', self.take(8))[0] ^ (_mask64(key) if key else 0)

    def number(self, key):
        value = 0
        while True:
            byte = self.byte(key)
            value = (value << 7) | (byte & 127)
            if not byte & 128:
                return value

    def string(self, key):
        length = self.number(key)
        raw = bytes(b ^ (key & 0xFF) for b in self.take(length))
        return raw.decode('utf-8')


def read(path):
    data = open(path, 'rb').read()
    reader = _Reader(data)
    archive = Archive()
    archive.header = reader.take(14)
    if archive.header[:4] != b'CAFF':
        raise ValueError(f'{path}: not a CAFF archive')
    archive.key = struct.unpack('>i', reader.take(4))[0]
    archive.reserved = reader.take(8)
    archive.preview = reader.take(28)
    key = archive.key
    count = reader.int32(key)
    table = []
    for _ in range(count):
        entry_path = reader.string(key)
        tag = reader.string(key)
        start = reader.int64(key) & 0xFFFFFFFF
        size = reader.int32(key)
        obfuscated = bool(reader.byte(key))
        compression = reader.byte(key)
        reader.take(8)
        table.append((entry_path, tag, start, size, obfuscated, compression))
    for entry_path, tag, start, size, obfuscated, compression in table:
        stored = data[start:start + size]
        if obfuscated:
            stored = bytes(bytearray(stored).translate(bytes(i ^ (key & 0xFF) for i in range(256))))
        if compression != COMPRESS_RAW:
            stored = _unzip_single(stored)
        archive.entries.append(Entry(entry_path, tag, stored, obfuscated, compression))
    return archive


def _unzip_single(blob):
    """Inflates a one-member zip stream.

    The Editor writes only a local file header, the raw DEFLATE data and a data descriptor - no
    central directory - so the zipfile module cannot open it.  The DEFLATE stream ends itself.
    """
    if blob[:4] != b'PK':
        raise ValueError('compressed entry does not start with a zip local file header')
    name_length, extra_length = struct.unpack('<HH', blob[26:30])
    return zlib.decompressobj(-15).decompress(blob[30 + name_length + extra_length:])


def _zip_single(content):
    """Deflates into the same partial zip stream the Editor writes (see _unzip_single)."""
    compressor = zlib.compressobj(6, zlib.DEFLATED, -15)
    deflated = compressor.compress(content) + compressor.flush()
    now = time.localtime()
    dos_time = (now.tm_hour << 11) | (now.tm_min << 5) | (now.tm_sec // 2)
    dos_date = ((now.tm_year - 1980) << 9) | (now.tm_mon << 5) | now.tm_mday
    name = b'contents'
    # version 2.0, flags 0x0808 (sizes in the data descriptor + UTF-8 name), method 8
    header = struct.pack('<4sHHHHHIIIHH', b'PK', 20, 0x0808, 8, dos_time, dos_date, 0, 0, 0, len(name), 0)
    descriptor = struct.pack('<4sIII', b'PK', zlib.crc32(content) & 0xFFFFFFFF, len(deflated), len(content))
    return header + name + deflated + descriptor


def _number(value):
    groups = [value & 127]
    value >>= 7
    while value:
        groups.append((value & 127) | 128)
        value >>= 7
    return bytes(reversed(groups))


def write(archive, path):
    key = archive.key
    low = key & 0xFF
    xor_table = bytes(i ^ low for i in range(256))

    def xor_bytes(raw):
        return bytes(bytearray(raw).translate(xor_table))

    def int32(value):
        return struct.pack('>I', (value ^ key) & 0xFFFFFFFF)

    def int64(value):
        return struct.pack('>Q', (value ^ _mask64(key)) & 0xFFFFFFFFFFFFFFFF)

    def string(text):
        raw = text.encode('utf-8')
        return xor_bytes(_number(len(raw)) + raw)

    stored_list = []
    for entry in archive.entries:
        stored = entry.data
        if entry.compression != COMPRESS_RAW:
            stored = _zip_single(stored)
        stored_list.append(stored)

    head = bytearray()
    head += archive.header or (b'CAFF' + bytes(3) + b'----' + bytes(3))
    head += struct.pack('>i', key)
    head += archive.reserved
    head += archive.preview or (bytes([127, 127]) + bytes(26))
    head += int32(len(archive.entries))
    table_size = sum(len(string(e.path)) + len(string(e.tag)) + 8 + 4 + 1 + 1 + 8 for e in archive.entries)
    position = len(head) + table_size
    table = bytearray()
    body = bytearray()
    for entry, stored in zip(archive.entries, stored_list):
        table += string(entry.path) + string(entry.tag)
        table += int64(position) + int32(len(stored))
        table += bytes([(1 if entry.obfuscated else 0) ^ low, entry.compression ^ low]) + bytes(8)
        body += xor_bytes(stored) if entry.obfuscated else stored
        position += len(stored)
    with open(path, 'wb') as handle:
        handle.write(bytes(head) + bytes(table) + bytes(body) + bytes([98, 99]))


def main(arguments):
    command = arguments[0]
    if command == 'list':
        archive = read(arguments[1])
        print(f'key={archive.key} entries={len(archive.entries)}')
        for entry in archive.entries:
            print(f'{entry.path:28s} tag={entry.tag:10s} size={len(entry.data):9d} compression={entry.compression} obfuscated={entry.obfuscated}')
    elif command == 'unpack':
        archive = read(arguments[1])
        os.makedirs(arguments[2], exist_ok=True)
        for entry in archive.entries:
            open(os.path.join(arguments[2], entry.path), 'wb').write(entry.data)
        print(f'unpacked {len(archive.entries)} entries to {arguments[2]}')
    elif command == 'repack':
        archive = read(arguments[1])
        archive.get('main.xml').data = open(arguments[2], 'rb').read()
        write(archive, arguments[3])
        print(f'wrote {arguments[3]}')
    else:
        print(__doc__)


if __name__ == '__main__':
    main(sys.argv[1:])
