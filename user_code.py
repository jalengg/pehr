import lxml.etree as etree

def canonicalize_xml(xml_str):
    """
    xml_str: String representation of the XML instance

    Use the lxml library to canonicalize xml_str and
    return the canonical xml as a UTF-8 encoded string
    """
    return etree.canonicalize(xml_data=xml_str)
