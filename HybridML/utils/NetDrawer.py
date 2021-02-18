import networkx as nx
import matplotlib.pyplot as plt


class NetDrawer:
    def draw_nets(self, nets):
        for net in nets:
            self.draw_net(net)

    def draw_net(self, net):

        G = nx.DiGraph()

        neural_nodes = [node.id for node in net.nodes]
        nodes = []
        nodes.extend(neural_nodes)
        data_points = [node for node in net.data_points.keys()]
        nodes.extend(data_points)
        net_nodes = ["Inputs", "Outputs"]
        nodes.extend(net_nodes)
        G.add_nodes_from(nodes)

        def edge(a, b, weight):
            return (a, b, {"weight": weight})

        edges = []
        for node in net.data_points.values():
            if node.input_node is not None:
                edges.append(edge(node.input_node.id, node.id, 0.5))
            for s in node.output_nodes:
                edges.append(edge(node.id, s.id, 1))

        net_outputs = [node for node in net.data_points.values() if len(node.output_nodes) == 0]

        inputs = [edge("Inputs", input.id, 0.5) for input in net.inputs]
        outputs = [edge(output.id, "Outputs", 2) for output in net_outputs]

        edges.extend(inputs)
        edges.extend(outputs)
        G.add_edges_from(edges)

        init = {"Inputs": (0, -2), "Outputs": (0, 2)}
        for input in net.inputs:
            init[input.id] = (0, -1)

        pos = nx.spring_layout(G, pos=init, weight="weight")  # positions for all nodes

        labels = {"Inputs": "Inputs", "Outputs": "Outputs"}
        nnnodes = {node: node for node in nodes}
        datapoints = {point.id: f"{point.id}:{point.size}" for point in net.data_points.values()}

        labels = {**nnnodes, **datapoints, **labels}

        nx.draw_networkx_nodes(G, pos, nodelist=neural_nodes, node_shape="s", node_color="r")
        nx.draw_networkx_nodes(G, pos, nodelist=data_points, node_shape="o", node_color="g")
        nx.draw_networkx_nodes(G, pos, nodelist=net_nodes, node_shape="v", node_color="y")
        nx.draw_networkx_labels(G, pos, labels=labels)

        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color="b", alpha=0.5)

        # plt.savefig("simple_path.png")  # save as png
        plt.show()  # display
