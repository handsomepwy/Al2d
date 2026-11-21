"""Print the deformer tree of an unpacked .cmo3 with each object's coordinate space and value range.

Usage: python cmo3_tree.py <main.xml>
"""
import sys

from al2d import cmo3xml


def main(path):
    doc = cmo3xml.Doc(path)
    guid_owner = {}   # guid element -> (kind, element)
    kinds = (('CArtMeshSource', 'mesh'), ('CWarpDeformerSource', 'warp'), ('CRotationDeformerSource', 'rot'))
    objects = []
    for tag, kind in kinds:
        for element in doc.definitions(tag):
            guid = doc.find_field(element, 'guid')
            guid_owner[guid] = (kind, element)
            objects.append((kind, element))

    def name_of(element):
        return (doc.find_field(element, 'localName').text or '')

    def coord_text(coord):
        if coord is None:
            return '?'
        return ' '.join(f'{k}={v}' for k, v in coord.attrib.items() if not k.startswith('xs.')) or \
               ' '.join((c.get('xs.n') or c.tag) + '=' + (c.text or '') for c in coord)

    rows = []
    for kind, element in objects:
        target = doc.find_field(element, 'targetDeformerGuid')
        parent = guid_owner.get(target)
        parent_text = f'{parent[0]}:{name_of(parent[1])}' if parent else f'ROOT({(target.get("note") if target is not None else None)})'
        forms = doc.field(element, 'keyforms')
        form_list = list(forms) if forms is not None else []
        details = ''
        if form_list:
            form = doc.deref(form_list[0])
            coord = doc.find_field(form, 'coordType')
            positions = doc.field(form, 'positions')
            if positions is not None:
                values = cmo3xml.floats(positions)
                xs, ys = values[0::2], values[1::2]
                details = f'coord[{coord_text(coord)}] x {min(xs):.3f}..{max(xs):.3f} y {min(ys):.3f}..{max(ys):.3f}'
            else:
                extra = ' '.join(f'{(c.get("xs.n"))}={c.text}' for c in form if c.tag in ('f', 'i') )
                details = f'coord[{coord_text(coord)}] {extra}'
        if kind == 'warp':
            details += f' grid {doc.field(element, "col").text}x{doc.field(element, "row").text}'
        rows.append(f'{kind:5s} {name_of(element)[:26]:26s} parent={parent_text[:34]:34s} keyforms={len(form_list):2d} {details}')
    print('\n'.join(rows))


if __name__ == '__main__':
    main(sys.argv[1])
