"""Compare the field layout of every object class in two cmo3 main.xml files.

    python schema_check.py <reference main.xml> <candidate main.xml>

For each element tag, the "layout" is the ordered list of its children's (tag, xs.n) pairs plus
its own attribute names.  A layout in the candidate that never occurs in the reference is
reported: it means we wrote an object differently from how the Editor writes it.
List containers are skipped (their children vary by nature).
"""
import sys
import xml.etree.ElementTree as ET

CONTAINERS = {'carray_list', 'array_list', 'hash_map', 'linked_map', 'linked_set', 'entry', 'shared', 'main', 'root'}
IGNORED_ATTRIBUTES = {'xs.id', 'xs.idx', 'xs.n', 'note'}


def layouts(path):
    found = {}
    for element in ET.parse(path).getroot().iter():
        if element.tag in CONTAINERS or 'xs.ref' in element.attrib:
            continue
        attributes = tuple(sorted(k for k in element.attrib if k not in IGNORED_ATTRIBUTES))
        children = tuple((child.tag, child.get('xs.n')) for child in element)
        found.setdefault(element.tag, {}).setdefault((attributes, children), 0)
        found[element.tag][(attributes, children)] += 1
    return found


def main():
    reference, candidate = layouts(sys.argv[1]), layouts(sys.argv[2])
    problems = 0
    for tag, variants in sorted(candidate.items()):
        if tag not in reference:
            print(f'{tag}: class not present in the reference')
            problems += 1
            continue
        for (attributes, children), count in variants.items():
            if (attributes, children) in reference[tag]:
                continue
            problems += 1
            print(f'{tag}: layout used {count}x is not in the reference')
            print(f'    ours      attrs={attributes} children={children}')
            closest = max(reference[tag], key=lambda v: len(set(v[1]) & set(children)) - len(set(v[1]) ^ set(children)))
            print(f'    reference attrs={closest[0]} children={closest[1]}')
    unused = sorted(set(reference) - set(candidate))
    print(f'{problems} layouts differ; classes only in the reference: {unused}')


if __name__ == '__main__':
    main()
