"""Helpers for reading a .cmo3 main.xml (as unpacked by Umamo's `extract`)."""
import xml.etree.ElementTree as ET


class Doc:
    def __init__(self, path):
        self.root = ET.parse(path).getroot()
        self.by_id = {}
        self.parent = {}
        for element in self.root.iter():
            if element.get('xs.id'):
                self.by_id[element.get('xs.id')] = element
            for child in element:
                self.parent[child] = element

    def deref(self, element):
        """Follows an xs.ref to the element that carries the xs.id."""
        if element is not None and element.get('xs.ref'):
            return self.by_id[element.get('xs.ref')]
        return element

    def field(self, element, name):
        """Direct child with xs.n == name (dereferenced)."""
        for child in element:
            if child.get('xs.n') == name:
                return self.deref(child)
        return None

    def find_field(self, element, name):
        """First descendant with xs.n == name, searching through `super` chains only."""
        hit = self.field(element, name)
        if hit is not None:
            return hit
        parent = self.field(element, 'super')
        return self.find_field(parent, name) if parent is not None else None

    def definitions(self, tag):
        """Elements of this tag that are definitions (not xs.ref pointers)."""
        return [e for e in self.root.iter(tag) if not e.get('xs.ref') and len(e)]


def floats(element):
    return [float(x) for x in (element.text or '').split()]
