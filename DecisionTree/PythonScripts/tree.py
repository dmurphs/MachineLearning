class Tree:
    def __init__(self,class_name=None,children={},decision_attribute=None):
        self.class_name = class_name
        self.children = children
        self.decision_attribute = decision_attribute

    def __str__(self):
        return 'class: %s, decision_attribute: %s, num_children: %i' %(self.class_name, self.decision_attribute, len(self.children))

    def add_child(self,attribute_val,child):
        self.children[attribute_val] = child

    def set_class_name(self,name):
        self.class_name = name

    def set_decision_attribute(self,attr):
        self.decision_attribute = attr
