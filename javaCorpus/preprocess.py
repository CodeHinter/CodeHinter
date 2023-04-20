import javalang
import re
import sys

class Prepossess:
    def __init__(self, file_path, output_file):
        self.file_path = file_path
        self.output = open(output_file, "w")

    def run(self):
        with open(self.file_path, encoding='utf-8') as f:
            input = f.read()

            split_code = re.findall(r'<s>([\s\S]*?)</s>', input)
            for code in split_code:
                self.print_result(code)
                self.output.write("=\n")

    def print_result(self, code):
        try:
            tree = javalang.parse.parse(code)
            for path, node in tree:
                if node.position is not None:
                    self.print_ast(node)

        except javalang.parser.JavaSyntaxError as e:
            print("Error parsing Java code:")
            print(code)
            # loop over the syntax errors and print the error messages
            # for error in e.errors:
            #     print("Exception message: "+ sys.argv[1], error)

    def print_ast(self, node, visited=None, depth=0):
        if visited is None:
            visited = set()
        indent = "  " * depth
        if id(node) in visited:
            return
        visited.add(id(node))
        if hasattr(node, "position") and node.position is not None:
            position = node.position
            if type(node).__name__ != "Literal" and type(node).__name__ != "ClassReference" \
            and type(node).__name__ != "MemberReference" and type(node).__name__ != "This":
                # self.output.write(f"{indent}{type(node).__name__} {position}\n")
                self.output.write(f"{indent}{type(node).__name__}\n")
        # else:
        #     self.output.write(f"{indent}{type(node).__name__}\n")

        if isinstance(node, javalang.ast.Node):
            # if the node is a container type, iterate over its children and call print_ast() recursively for each child
            for child_name, child_node in node.filter(javalang.ast.Node):
                self.print_ast(child_node, visited, depth + 1)

            # if the node has a body attribute, call print_ast() recursively for each element of the body
            if hasattr(node, "body") and node.body is not None:
                for i in node.body:
                    self.print_ast(i, visited, depth + 1)
                node.body = [i for i in node.body if id(i) not in visited]

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py arg1 arg2")
        sys.exit(1)
    sys.setrecursionlimit(20000)
    pre = Prepossess(sys.argv[1], sys.argv[2])
    pre.run()

