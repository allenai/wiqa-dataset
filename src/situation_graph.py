# Tasks:
# input = graph in some format
# output1 = graph in json format
# output2 = tf_ques_given_a_json_record
# output3 = generate explanation candidates for_a_tf_ques

import copy
import enum
import json
from typing import List


def strip_special_char(a_string):
    return "".join([x for x in a_string if (ord('a') <= ord(x) <= ord('z')) or (ord('A') <= ord(x) <= ord('Z'))]).lower()


class SituationNode:
    def __init__(self, node_id: str,
                 the_groundings: [],
                 is_decision_node: bool = False,
                 node_semantics: str = ""):
        '''
        :param node_id e.g., "VX"
        :param node_semantics e.g., for A/D node, semantics is "accelerates", "decelerates"; 
                                or, x/y node can be "causal" or u,v,w,z "indirect". 
        :param the_groundings: is an array of groundings.
        :param is_decision_node: e.g., Accelerates or Decelerates nodes are decision nodes currently 
        '''
        self.id = node_id
        self.node_semantics = node_semantics
        self.groundings = [t for t in the_groundings]  # we need a copy and not a reference of the supplied array.
        self.is_decision_node = is_decision_node

    def join_groundings(self, separator):
        if not self.groundings:
            return ""
        return separator.join(self.groundings)

    def remove_grounding(self, the_grounding_to_remove):
        try:
            self.groundings.remove(the_grounding_to_remove)
        except ValueError:
            # Continue, do not stop
            print(f"Warning: Removal of a non-existent grounding: '{the_grounding_to_remove}' from node {self.id}")

    def get_grounding_pairs(self):
        if not self.groundings or len(self.groundings) < 2:
            return []
        return zip(*[self.groundings[i:] for i in range(2)])

    def is_empty(self):
        return not self.groundings or len(self.groundings) == 0

    def __repr__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def get_specific_grounding(self, specific_grounding=None):
        '''
        :param specific_grounding: if set to None, then all groundings are used {source_node.groundings} otherwise the string supplied. 
        :return: 
        '''
        return f"{self.groundings}" if not specific_grounding else specific_grounding

    def get_first_grounding(self):
        return "" if not self.groundings or len(self.groundings) < 1 else self.groundings[0]


class SituationLabel(str, enum.Enum):
    NOT_RESULTS_IN ="NOT_RESULTS_IN"
    RESULTS_IN = "RESULTS_IN"
    NO_EFFECT = "NO_EFFECT"
    MARKED_NOISE = "MARKED_NOISE"

    def get_sign(self):
        if self == SituationLabel.RESULTS_IN:
            return '+'
        elif self == SituationLabel.NOT_RESULTS_IN:
            return '-'
        else:
            return '.'

    def get_sign_str(self):
        if self == SituationLabel.RESULTS_IN:
            return 'MORE'
        elif self == SituationLabel.NOT_RESULTS_IN:
            return 'LESS'
        else:
            return 'NOEFFECT'

    def as_less_more(self):
        if self == SituationLabel.RESULTS_IN:
            return 'more'
        elif self == SituationLabel.NOT_RESULTS_IN:
            return 'less'
        else:
            return 'no_effect'

    def get_nickname(self):
        if self == SituationLabel.RESULTS_IN:
            return 'a'
        elif self == SituationLabel.NOT_RESULTS_IN:
            return 'd'
        else:
            return '-'

    def get_opposite_label(self):
        if self == SituationLabel.RESULTS_IN:
            return SituationLabel.NOT_RESULTS_IN
        elif self == SituationLabel.NOT_RESULTS_IN:
            return SituationLabel.RESULTS_IN
        else:
            return self

    def get_emnlp_test_choice(self):
        if self == SituationLabel.RESULTS_IN:
            return 'A'
        elif self == SituationLabel.NOT_RESULTS_IN:
            return 'B'
        else:
            return 'C'

    @staticmethod
    def from_str(sl):
        if not sl:
            raise ValueError(
                f"({sl}) is not a valid Enum SituationLabel")
        sl = sl.lower().replace('_', ' ').strip()

        if sl in ['-', 'not results in', 'opposite', 'opp', 'results in opp', 'opp effect', 'opposite effect', 'less']:
            return SituationLabel.NOT_RESULTS_IN
        elif sl in ['+', 'positive', 'results in', 'correct', 'correct effect', 'more']:
            return SituationLabel.RESULTS_IN
        elif sl in ['.', 'none', 'no effect']:
            return SituationLabel.NO_EFFECT
        else:
            print(f"WARNING: ({sl}) is not a valid Enum SituationLabel")
            return SituationLabel.MARKED_NOISE

    def to_readable_str(self):
        if self == SituationLabel.RESULTS_IN:
            return 'RESULTS_IN'
        elif self == SituationLabel.NOT_RESULTS_IN:
            return 'RESULTS_IN_OPP'
        else:
            return 'NO_EFFECT'

    def to_json(self):
        return self.name


class SituationEdge:
    def __init__(self, from_node: SituationNode, to_node: SituationNode, label: SituationLabel, wt=1.0):
        self.from_node = from_node
        self.to_node = to_node
        self.wt = wt
        self.label = label
        self.id = (from_node.id, to_node.id)
        self.is_noise = False

    def mark_noise(self):
        # If marked as noise, then we will drop this edge completely
        self.is_noise = True

    def reset_label_to(self, new_label: SituationLabel):
        # If marked as noise, then we will drop this edge completely
        self.label = new_label

    def ground_edge(self):
        grounded_edges = []
        # node A (A1, A2) => node B (B1, B2)
        for s1 in self.from_node.groundings:
            for s2 in self.to_node.groundings:
                grounded_edges.append({"from_node": s1, "to_node": s2, "label": self.label.name})
        return grounded_edges

    def __repr__(self):
        return self.from_node.id + "-" + self.to_node.id


class SituationGraph:
    # graph structure is a json of edges and their labels.
    # e.g., [(V,X,not)
    def __init__(self, situation_nodes: List[SituationNode], situation_edges: List[SituationEdge],
                 other_properties: dict):
        '''
        :param situation_nodes: an array of situation nodes.
        :param other_properties: e.g., other_graph_provenance, 
                                 or, para_outcome,
                                 or, graph_provenance: paraid__blocknum__paraprompt etc.
        '''
        self.nodes = situation_nodes
        self.situation_edges = situation_edges
        self.other_properties = {k: v for k, v in other_properties.items()}
        self.cleanup()

    def cleanup(self, remove_no_effects=True):
        """
        This function can also be called at a later point when vetting data is available on a situation graph.
        @:param remove_no_effects : unless we extend our graphs to contain no effect nodes, the default should be
        True
        :return: 
        """
        self.remove_empty_nodes()
        edges_to_remove = [e for e in self.situation_edges
                           if e.is_noise or e.label == SituationLabel.MARKED_NOISE or (
                                   remove_no_effects and e.label == SituationLabel.NO_EFFECT)]
        for e in edges_to_remove:
            self.remove_an_edge(edge_to_remove=e)

    def copy(self):
        copied = SituationGraph(
            situation_nodes=copy.deepcopy(self.nodes),
            situation_edges=copy.deepcopy(self.situation_edges),
            other_properties=copy.deepcopy(self.other_properties)
        )
        copied.cleanup()
        return copied

    def get_empty_nodes(self):
        return [n for n in self.nodes if n.is_empty()]

    def remove_empty_nodes(self):
        nodes_to_remove = self.get_empty_nodes()
        for cand_node in nodes_to_remove:
            self.remove_a_node(node_to_remove=cand_node)

    def get_all_node_grounding_pairs(self):
        lists = []
        for x in self.nodes:
            pairs = x.get_grounding_pairs()
            if pairs:
                lists.extend(pairs)
        return lists

    def remove_a_node(self, node_to_remove):
        if not node_to_remove:
            return
        try:
            self.nodes.remove(node_to_remove)
            # also remove all edges it occurs in.
            edges_to_remove = [e for e in self.situation_edges
                               if e.from_node.id == node_to_remove.id or e.to_node.id == node_to_remove.id]
            for e in edges_to_remove:
                self.remove_an_edge(edge_to_remove=e)
        except ValueError:
            print(f"Warning: Removal of a non-existent node: '{node_to_remove}'")

    def get_exogenous_nodes(self, exogenous_ids=["Z", "V"]):
        return [self.lookup_node(node_id=x) for x in exogenous_ids]

    def remove_an_edge(self, edge_to_remove):
        if not edge_to_remove:
            return
        try:
            self.situation_edges.remove(edge_to_remove)
        except ValueError:
            print(f"Warning: Removal of a non-existent edge: '{edge_to_remove}'")

    def lookup_node(self, node_id) -> SituationNode:
        for t in self.nodes:
            if t.id == node_id:
                return t
        return None

    def lookup_node_from_grounding(self, grounding) -> SituationNode:
        for t in self.nodes:
            # There is a very small chance that a grounding in two different nodes is same.
            # For those cases, this function just picks the nodes that comes first in the self.nodes array
            if grounding in t.groundings:
                return t

        # Search unsuccessful, so try removing lowercasing/uppercasing and tokenization mismatches (drop special chars)
        stripped_grounding = strip_special_char(a_string=grounding)
        if not stripped_grounding:
            return None
        for t in self.nodes:
            if stripped_grounding in [strip_special_char(x) for x in t.groundings]:
                return t

        return None

    def lookup_edge(self, edge_source_node, edge_target_node) -> SituationEdge:
        for t in self.situation_edges:
            if t.from_node == edge_source_node and t.to_node == edge_target_node:
                return t
        return None

    def speak_path(self, path, path_label, as_symbols=False, specific_grounding=None):
        if not path or len(path) < 1:
            return ""
        return self.speak_path_with_symbols(path=path, path_label=path_label, specific_grounding=specific_grounding) \
            if as_symbols else self.speak_path_in_sentences(path=path, path_label=path_label,
                                                            specific_grounding=specific_grounding)

    def construct_path_from_str_arr(self, str_node_arr):
        path = [self.lookup_node(node_str.strip()) for node_str in str_node_arr]
        path_ok = True
        for node in path:
            if node is None or node.is_empty():
                path_ok = False
        return path if path_ok else None

    def get_graph_id(self):
        return self.other_properties["graph_id"]

    def speak_path_with_symbols(self, path, path_label, specific_grounding=None):
        '''
        
        :param path: array of Situation nodes e.g. V, X, A
        :param path_label: array of path_len - 1 nodes, e.g., [RESULTS_IN, NOT_RESULTS_IN]
        :return: 
        '''
        return f"{path[0].get_specific_grounding(specific_grounding)}->" + \
               ("->".join([l.name + "->" + f"{p.get_specific_grounding(specific_grounding)}"
                           for p, l in zip(path[1:], path_label)]))

    def speak_path_in_sentences(self, path, path_label, specific_grounding=None):
        '''
        Symbolic forms are very confusing:

        ```A not B not C
        A =/> B =/> C``` 
        
        So instead express it pairwise as multiple sentences (each explanation answer option can be list of sentences):
        
        ```If A happens, then B will not happen. 
        If B happens, then C will not happen.```
        
        :param path: array of Situation nodes e.g. V, X, A
        :param path_label: array of path_len - 1 nodes, e.g., [RESULTS_IN, NOT_RESULTS_IN]
        :return: 
        '''

        sentences = []
        for edge_num, (n1, n2) in enumerate(zip(path, path[1:])):
            speak_label = "will" if path_label[edge_num] == SituationLabel.RESULTS_IN else "will not"
            sentences.append(f"If {n1.get_specific_grounding(specific_grounding)} happens, then "
                             f"{n2.get_specific_grounding(specific_grounding)} {speak_label} happen")

        return ". ".join(sentences).strip()  # strip removes last space
        # return f"{path[0].get_specific_grounding(specific_grounding)}->" + \
        #        ("->".join([l.name + "->" + f"{p.get_specific_grounding(specific_grounding)}"
        #                    for p, l in zip(path[1:], path_label)]))

    def all_labels_in(self, path):
        return [self.lookup_edge(src, dest).label if self.lookup_edge(src,
                                                                      dest) is not None else SituationLabel.NOT_RESULTS_IN
                for src, dest in zip(path, path[1:])]
        # raw_path_labels = [self.lookup_edge(src, dest).label for src, dest in zip(path, path[1:])]
        # fixed_all_labels = raw_path_labels[:-1]
        # # FIXME this is possibly a bug and needs to be fixed when end label in path != decision node.
        # # and if y=/>a that means flip the inherent label.
        # # end node in the path is {accelerate/ decelerate}
        # if path[-1].is_decision_node and not self.other_properties.get("is_positive_outcm", True):
        #     fixed_all_labels.append(raw_path_labels[0].get_opposite_label())
        # else:
        #     fixed_all_labels.append(raw_path_labels[-1])
        # # fixed_all_labels.append(self.end_label_for_path(path=path))
        # return fixed_all_labels

    # one process per html page (9 situation graphs at most).
    # For perl: https://metacpan.org/pod/distribution/Graph-Easy/bin/graph-easy
    # For python, see: https://pypi.org/project/graphviz/
    # For javascript, see: http://www.webgraphviz.com/
    @staticmethod
    def to_html(graphviz_data_map, html_title, html_outfile_path=None):
        '''
        # <html>
        # <head>
        #     <title>Paragraph 1</title>
        #     <script src="http://people.mpi-inf.mpg.de/~ntandon/resources/viz.js"></script>
        # </head>
        #
        # <body>
        # <p> html_page_top_str </p>
        # <div id="visu1"> </div>
        # <div id="visu2"> </div>
        #
        # <script>
        #     function UpdateGraphviz() {
        #            var data = 'digraph G { "Welcome" -> "To" "To" -> "Web" "To" -> "GraphViz"}';
        #            var svg = Viz(data, "svg");
        #            div = document.getElementById("visu1");
        #            div.innerHTML = "<hr>" + svg;
        #            // svg_div.html("<hr>" + svg);
        #
        #            var data2 = 'digraph G2 { "Welcome2" -> "To2" "To2" -> "Web2" "To2" -> "GraphViz2"}';
        #            var svg2 = Viz(data2, "svg");
        #            div2 = document.getElementById("visu2");
        #            div2.innerHTML = "<hr>" + svg2;
        #          }
        # </script>
        #
        # <script>
        #     UpdateGraphviz();
        # </script>
        #
        # </body>
        # </html>
        :param graphviz_data_map: 
        :param html_title: 
        :param html_outfile_path: 
        :return: 
        '''
        htmls = list()
        assert graphviz_data_map is not None and len(graphviz_data_map) > 0, \
            f"Input to graphviz is empty. This function call must be fixed: " \
                f"\nSituationGraph.to_html(graphviz_data_map=empty_array, html_title={html_title})"
        htmls.append(f"<html> \n <head> \n<meta charset=\"UTF-8\"> \n <title> {html_title} </title>\n")
        htmls.append('<script src="http://people.mpi-inf.mpg.de/~ntandon/resources/viz.js"></script>\n</head>\n')
        htmls.append(f"<body>\n")
        htmls.append(f"\n<br><br><hl>")
        htmls.append("\n".join([f'\n\n<p id="desc{x_idx}"></p>\n<div id="visu{x_idx}"> </div>' for x_idx, _ in
                                enumerate(graphviz_data_map.keys())]))
        htmls.append(f"\n<script>\nfunction graphviz_input()" + "{\n")
        htmls.append("\n".join([f'\nvar svg_{x_idx} = Viz(\'{x_val}\', "svg"); '
                                f'\ndocument.getElementById("visu{x_idx}").innerHTML="<hr>" + svg_{x_idx};'
                                f'\ndocument.getElementById("desc{x_idx}").innerHTML="<hr>" + "{x_desc}";\n\n'
                                for x_idx, (x_desc, x_val) in enumerate(graphviz_data_map.items())]))
        htmls.append("\n}\n" + f"\n</script>\n\n<script>graphviz_input();</script>\n\n</body>\n</html>")
        html_str = "\n".join(htmls)
        if html_outfile_path is not None:
            html_outfile = open(html_outfile_path, 'w')
            html_outfile.write(html_str)
            html_outfile.close()
        return html_str

    def as_graphviz(self, statement_separator):
        '''
        :param statement_separator either "\n" (human readable) or "\t" (for html)
        ## Sample digraph
        ## color: "implies" : green, "not implies": red.
        ## node[style=filled, color=cornflowerblue, fontcolor=white, fontsize=10, fontname='Helvetica']
        ## edge[arrowhead=vee, arrowtail=inv, arrowsize=.7, color=maroon, fontsize=10, fontcolor=navy]
        # digraph G {
        # "X" -> "Y"[label = "implies"]
        # "X" -> "W"[label = "not implies"]
        # "U" -> "Y"[label = "not implies"]
        # "The sun was not in the sky\nThe sun was in the sky" -> "X"[label = "implies"]
        # "V" -> "X"[label = "not implies"]
        # "Y" -> "A"[label = "implies"]
        # "Y" -> "D"[label = "not implies"]
        # "W" -> "A"[label = "not implies"]
        # "W" -> "D"[label = "implies"]
        # }
        :return: 
        '''
        g = list()
        printed_newline = "________"  # "\\\n"
        g.append("digraph G {")
        for e in self.situation_edges:
            edge_color = "green" if e.label == SituationLabel.RESULTS_IN else (
                "red" if e.label == SituationLabel.NOT_RESULTS_IN else "yellow")
            node1 = e.from_node.join_groundings(separator=printed_newline)
            node2 = e.to_node.join_groundings(separator=printed_newline)
            edge = f"\"{node1}\" -> \"{node2}\" [color={edge_color}]"
            g.append(edge.replace(printed_newline, ("\\\\" + "n")))
        g.append("}")
        return statement_separator.join(g)

    def as_graphviz_with_labels(self, statement_separator):
        """ This is like as_graphviz, but with labels for each node. """
        g = list()
        printed_newline = "________"  # "\\\n"
        g.append("digraph G {")
        for e in self.situation_edges:
            edge_color = "green" if e.label == SituationLabel.RESULTS_IN else (
                "red" if e.label == SituationLabel.NOT_RESULTS_IN else "yellow")
            node1 = "Node " + e.from_node.id + printed_newline + e.from_node.join_groundings(separator=printed_newline)
            node2 = "Node " + e.to_node.id + printed_newline + e.to_node.join_groundings(separator=printed_newline)
            edge = f"\"{node1}\" -> \"{node2}\" [color={edge_color}]"
            g.append(edge.replace(printed_newline, ("\\\\" + "n")))
        g.append("}")
        return statement_separator.join(g)

    def end_label_for_path(self, path):
        # Suppose, X ==> Y
        #       X =/=> W
        #       U =/=> Y
        #       Z ==> X
        #       V =/=> X
        # if we encounter even number of not's then answer is True else False
        # This will work for all cases even the intermediate paths V =/=> X =/=> W
        raw_path_labels = [self.lookup_edge(src, dest).label for src, dest in zip(path, path[1:])]
        return SituationLabel.RESULTS_IN \
            if len([x for x in raw_path_labels if x == SituationLabel.NOT_RESULTS_IN]) % 2 == 0 \
            else SituationLabel.NOT_RESULTS_IN
        #
        #
        # num_negations = 0
        # for l in raw_path_labels:
        #     if l == SituationLabel.NOT_RESULTS_IN:
        #         num_negations += 1
        # if num_negations % 2 == 0:
        #     return SituationLabel.RESULTS_IN
        # return SituationLabel.NOT_RESULTS_IN

    # previous end_label_for_path implementation
    # def end_label_for_path_old(self, path):
    #     # Suppose, X ==> Y
    #     #       X =/=> W
    #     #       U =/=> Y
    #     #       Z ==> X
    #     #       V =/=> X
    #     # Then, V,X,W,A path ...
    #     #       not, not, not
    #     # so, if ever encounter "not" then answer is "not"
    #     # Other example, V,X,Y,A
    #     # so,
    #     raw_path_labels = [self.lookup_edge(src, dest).label for src, dest in zip(path, path[1:])]
    #     for l in raw_path_labels:
    #         if l == SituationLabel.NOT_RESULTS_IN:
    #             return SituationLabel.NOT_RESULTS_IN
    #     return SituationLabel.RESULTS_IN

    def nodes_from(self, source_node):
        return [t.to_node for t in self.situation_edges if t.from_node.id == source_node.id]

    def dfs_between(self, source_node, target_node, visited, current_path, all_paths, min_path_len):
        visited[source_node.id] = True
        current_path.append(source_node)

        if source_node.id == target_node.id:
            # We found one path to the target, record it.
            if len(current_path) >= min_path_len:  # x=>y=>z has path length 3, which is accepted if minlen is 3
                all_paths.append(current_path.copy())

        else:
            # We haven't arrived at the target node yet, keep searching.
            for e in self.nodes_from(source_node=source_node):
                if not visited[e.id]:
                    self.dfs_between(source_node=e,
                                     target_node=target_node,
                                     current_path=current_path,
                                     visited=visited,
                                     all_paths=all_paths,
                                     min_path_len=min_path_len
                                     )

        # target_node shouldn't be marked visited forever
        # current node may be part of another path so allow it to be visited again.
        visited[source_node.id] = False
        current_path.pop()  # remove current node.

    def paths_between(self, source_node, target_node, min_path_len):
        all_paths = []
        visited = {t.id: False for t in self.nodes}
        self.dfs_between(source_node=source_node,
                         target_node=target_node,
                         visited=visited,
                         current_path=[],
                         all_paths=all_paths,
                         min_path_len=min_path_len
                         )
        return all_paths

    # def distractor_paths(self, paths):
    #     invalid_paths = []
    #     valid_paths = self.paths_between(source_node=source_node, target_node=target_node, min_path_len=min_path_len)
    #
    #     # source nodes can never be decision nodes.
    #     query_source_nodes = [t for t in self.nodes if not t.is_decision_node]
    #     # target nodes may or may not be decision nodes.
    #     query_target_nodes = [t for t in self.nodes if not t.is_decision_node]
    #     for src in query_source_nodes:
    #         for tgt in query_target_nodes:
    #             if src != tgt:
    #                 paths_between = self.paths_between(source_node=src, target_node=tgt, min_path_len=min_path_len)
    #                 if not paths_between:
    #                     all_queries.append(paths_between)
    #     return invalid_paths

    def all_query_paths(self, min_path_len, target_node_has_to_be_decision_node, hardcoded_source_nodes_strs=None):
        '''
        
        :param hardcoded_source_nodes: 
        :param min_path_len: e.g. 3 would mean that Z=>X would be ignored but Z,,W 
        :param target_node_has_to_be_decision_node: if set to False, then both source and tgt nodes are internal nodes (and not A or D nodes)  
        :return: 
        '''
        all_queries = []
        # source nodes can never be decision nodes.
        cleaned_source_nodes = self.construct_path_from_str_arr(
            str_node_arr=hardcoded_source_nodes_strs) if hardcoded_source_nodes_strs else []
        if hardcoded_source_nodes_strs is not None and not cleaned_source_nodes:
            return all_queries
        query_source_nodes = cleaned_source_nodes or [t for t in self.nodes if not t.is_decision_node]
        # target nodes may or may not be decision nodes.
        query_target_nodes = [t for t in self.nodes if (
            (t.is_decision_node or not target_node_has_to_be_decision_node)
            # and t not in query_source_nodes
        )
                              ]
        for src in query_source_nodes:
            for tgt in query_target_nodes:
                if src.id != tgt.id:
                    paths_between = self.paths_between(source_node=src, target_node=tgt, min_path_len=min_path_len)
                    # valid_paths = self.get_valid_paths(cand_paths=paths_between, source_nodes=query_source_nodes)
                    valid_paths = self.get_valid_paths(cand_paths=paths_between, source_nodes=cleaned_source_nodes)
                    if valid_paths:
                        all_queries.append(valid_paths)
        return all_queries

    # given a situationgraph id we need (list of {s,o,p} dict).
    def get_grounded_edges(self):
        grounded_edges = []
        # node A (A1, A2) => node B (B1, B2)
        for edge in self.situation_edges:
            for edge_grounding in edge.ground_edge():
                grounded_edges.append(edge_grounding)
        return grounded_edges

    def to_json_v1(self):
        return json.dumps(self.to_struct_v1())

    def to_struct_v1(self):
        struct = {}

        for n in self.nodes:
            the_name = n.id
            if the_name == "A":
                the_name = "para_outcome_accelerate"
            if the_name == "D":
                the_name = "para_outcome_decelerate"
            struct[the_name] = n.groundings

        # (not used) "para_outcome": "oil formation",
        # (not used) "Y_is_outcome": ""
        # "Y_affects_outcome": "-",
        y_to_a_label = self.lookup_edge(edge_source_node=self.lookup_node(node_id="Y"),
                                        edge_target_node=self.lookup_node(node_id="A")).label
        struct["Y_affects_outcome"] = y_to_a_label.get_nickname()
        struct["paragraph"] = self.other_properties.get("paragraph", "")
        struct["prompt"] = self.other_properties.get("prompt", "")
        struct["para_id"] = self.other_properties.get("para_id", "")
        struct["Y_is_outcome"] = self.other_properties.get("y_is_outcome", "")

        return struct

    @staticmethod
    def from_json_v1(json_string: str):
        # decode the JSON string into a structure
        return SituationGraph.from_struct_v1(struct=json.loads(json_string))

    @staticmethod
    def from_struct_v1(struct: dict):
        '''
        
        :param struct (for data_version "v1"): a data structure that looks like this:
         {
           "para_id":"propara_pilot1_task12_p1.txt",
           "prompt":"How does oil form?",
           "paragraph":"Algae and plankton die. 
                       The dead algae and plankton end up part of sediment on a seafloor. 
                       The sediment breaks down. 
                       The bottom layers of sediment become compacted by pressure. 
                       Higher pressure causes the sediment to heat up. 
                       The heat causes chemical processes. 
                       The material becomes a liquid. 
                       Is known as oil. 
                       Oil moves up through rock.",
           "X":"pressure on sea floor increases",
           "Y":"sediment becomes hotter",
           "W":[
              "sediment becomes cooler"
           ],
           "U":[
        
           ],
           "Z":[
              "ocean levels rise",
              "more plankton then normal die"
           ],
           "V":[
              "oceans evaporate"
           ],
           "para_outcome":"oil formation",
           "para_outcome_accelerate":
              "More oil forms"
           ,
           "para_outcome_decelerate":
              "Less oil forms"
           ,
           "Y_affects_outcome":"-",
           "Y_is_outcome":""
        }
        
        # graph_version "v1" assumes that:
        #               X ==> Y 
        #               X =/=> W 
        #               U =/=> Y 
        #               Z ==> X 
        #               V =/=> X 
        #               W =/=> A 
        #               W ==> D 
        #               Y ==> A or Y ==> D  
        
        :return:
         SituationGraph object.
         
        '''

        # Fill an empty node because the graph structure is fixed.
        for expected_inner_node in ["Z", "V", "X", "Y", "W", "U"]:
            if expected_inner_node not in struct:
                struct[expected_inner_node] = []

        node_v = SituationNode(node_id="V", the_groundings=struct["V"])
        node_z = SituationNode(node_id="Z", the_groundings=struct["Z"])
        node_x = SituationNode(node_id="X",
                               the_groundings=struct["X"] if isinstance(struct["X"], list) else [
                                   struct["X"]])
        node_u = SituationNode(node_id="U", the_groundings=struct["U"])
        node_w = SituationNode(node_id="W", the_groundings=struct["W"])
        node_y = SituationNode(node_id="Y",
                               the_groundings=struct["Y"] if isinstance(struct["Y"], list) else [
                                   struct["Y"]])
        outcm_accelerates = "accelerates process" if "para_outcome_accelerate" not in struct or not struct[
            "para_outcome_accelerate"] else struct["para_outcome_accelerate"]
        outcm_decelerates = "decelerates process" if "para_outcome_decelerate" not in struct or not struct[
            "para_outcome_decelerate"] else struct["para_outcome_decelerate"]
        node_a = SituationNode(node_id="A",
                               the_groundings=outcm_accelerates if isinstance(outcm_accelerates, list) else [
                                   outcm_accelerates],
                               is_decision_node=True,
                               node_semantics="accelerates")
        node_d = SituationNode(node_id="D",
                               the_groundings=outcm_decelerates if isinstance(outcm_decelerates, list) else [
                                   outcm_decelerates],
                               is_decision_node=True,
                               node_semantics="decelerates")

        nodes = [node_v, node_z, node_x, node_u, node_w, node_y, node_a, node_d]

        #########################
        # BEGIN-hardcoding
        #########################
        outcm = "Y_affects_outcome"
        is_positive_outcm = True  # True if y=>a
        # If the following two were provided in input_json, then no pre-processing is needed
        ya_label = SituationLabel.MARKED_NOISE
        yd_label = SituationLabel.MARKED_NOISE
        if struct[outcm] == 'a' or struct[outcm] == True or struct[outcm] == 'more':
            ya_label = SituationLabel.RESULTS_IN
            yd_label = SituationLabel.NOT_RESULTS_IN
        elif struct[outcm] == 'd' or struct[outcm] == False or struct[outcm] == 'less':
            ya_label = SituationLabel.NOT_RESULTS_IN
            yd_label = SituationLabel.RESULTS_IN
            is_positive_outcm = False
        else:
            ya_label = SituationLabel.NO_EFFECT
            yd_label = SituationLabel.NO_EFFECT

        if ya_label == (SituationLabel.MARKED_NOISE or SituationLabel.NO_EFFECT) \
                or yd_label == (SituationLabel.MARKED_NOISE or SituationLabel.NO_EFFECT):
            return None
        #########################
        # END-hardcoding
        #########################

        edges = [
            SituationEdge(from_node=node_v, to_node=node_x, label=SituationLabel.NOT_RESULTS_IN),
            SituationEdge(from_node=node_z, to_node=node_x, label=SituationLabel.RESULTS_IN),
            SituationEdge(from_node=node_x, to_node=node_w, label=SituationLabel.NOT_RESULTS_IN),
            SituationEdge(from_node=node_x, to_node=node_y, label=SituationLabel.RESULTS_IN),
            SituationEdge(from_node=node_u, to_node=node_y, label=SituationLabel.NOT_RESULTS_IN),
            # SituationEdge(from_node=node_w, to_node=node_a, label=SituationLabel.NOT_RESULTS_IN),
            # SituationEdge(from_node=node_w, to_node=node_d, label=SituationLabel.RESULTS_IN),
            SituationEdge(from_node=node_w, to_node=node_a, label=yd_label),
            SituationEdge(from_node=node_w, to_node=node_d, label=ya_label),
            SituationEdge(from_node=node_y, to_node=node_a, label=ya_label),
            SituationEdge(from_node=node_y, to_node=node_d, label=yd_label)
        ]
        paragraph = struct["paragraph"]
        paragraph = paragraph.replace("\"", "'")

        return SituationGraph(
            situation_nodes=nodes,
            situation_edges=edges,
            other_properties={
                "para_id": struct["para_id"],
                "prompt": struct["prompt"],
                "paragraph": paragraph,
                "outcome": struct.get("para_outcome", ""),
                "y_is_outcome": struct["Y_is_outcome"],
                "is_cyclic_process": struct.get("is_cyclic_process", False),
                "is_positive_outcm": is_positive_outcm,
                "graph_id": struct.get("graph_id", "")
            }
        )

    def get_valid_paths(self, cand_paths, source_nodes):
        if not cand_paths:
            return []
        return [p for p in cand_paths if not self.is_any_source_node_in_path(path=p, sources_nodes=source_nodes)]

    def is_any_source_node_in_path(self, path, sources_nodes):
        '''
        
        :param path: [Z,X,Y,A]
        :param sources_nodes: [V,Z,U,Y]
        :return: in this example since Y is present in the non-starting node of the path so returns True
        '''
        source_nodes_ids = [x.id for x in sources_nodes]
        for p in path[1:]:
            if p.id in source_nodes_ids:
                return True
        return False
