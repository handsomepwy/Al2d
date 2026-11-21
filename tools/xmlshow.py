"""Print parts of a cmo3 main.xml as an indented outline (long lists capped).

  python xmlshow.py <main.xml> main [cap]             the <main> section
  python xmlshow.py <main.xml> tag <Tag> [n] [cap]    the n-th shared definition of a tag
  python xmlshow.py <main.xml> id <#123> [cap]        the element with that xs.id
  python xmlshow.py <main.xml> shared                 one line per shared object, in file order
"""
import sys
import xml.etree.ElementTree as ET

LISTS = ('carray_list', 'array_list', 'hash_map', 'linked_map', 'linked_set')


def show(element, depth, out, cap=3):
    attributes = ' '.join(f'{k}="{v}"' for k, v in element.attrib.items())
    text = (element.text or '').strip()
    if len(text) > 70:
        text = text[:70] + f'... ({len(text)} chars)'
    out.append('  ' * depth + f'<{element.tag} {attributes}> {text}')
    children = list(element)
    for index, child in enumerate(children):
        if element.tag in LISTS and index >= cap:
            out.append('  ' * (depth + 1) + f'... {len(children) - cap} more')
            break
        show(child, depth + 1, out, cap)


def main(arguments):
    root = ET.parse(arguments[0]).getroot()
    mode = arguments[1]
    out = []
    if mode == 'main':
        show(root.find('main'), 0, out, int(arguments[2]) if len(arguments) > 2 else 3)
    elif mode == 'tag':
        found = [e for e in root.find('shared') if e.tag == arguments[2]]
        index = int(arguments[3]) if len(arguments) > 3 else 0
        out.append(f'{len(found)} shared <{arguments[2]}> definitions; showing #{index}')
        show(found[index], 0, out, int(arguments[4]) if len(arguments) > 4 else 3)
    elif mode == 'id':
        for element in root.iter():
            if element.get('xs.id') == arguments[2]:
                show(element, 0, out, int(arguments[3]) if len(arguments) > 3 else 3)
                break
    elif mode == 'shared':
        for element in root.find('shared'):
            attributes = ' '.join(f'{k}="{v}"' for k, v in element.attrib.items() if k != 'xs.idx')
            out.append(f'{element.tag} {attributes}'[:150])
    sys.stdout.reconfigure(encoding='utf-8')
    print('\n'.join(out))


if __name__ == '__main__':
    main(sys.argv[1:])
