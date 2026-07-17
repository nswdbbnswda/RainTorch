#include <iostream>
#include <vector>
#include <unordered_set>
#include <functional>
#include <fstream>
using namespace std;

class DirectedGraph
{
private:
    vector<vector<int>> adj;
    unordered_set<int> node_set;

public:
    DirectedGraph(int cap = 50)
    {
        adj.resize(cap);
    }

    void add_node(int u)
    {
        node_set.insert(u);
    }

    void add_edge(int u, int v)
    {
        add_node(u);
        add_node(v);
        if (adj.size() < u) {
          std::cout << "adj.size:" << adj.size() << std::endl;
          adj.resize(u);
        }
        adj[u].push_back(v);
    }

    // 列出所有节点与所有边文本
    void list_all()
    {
        cout << "=== 所有节点 ===" << endl;
        for (int n : node_set)
        {
            cout << n << " ";
        }
        cout << "\n\n=== 全部有向边 u -> v ===" << endl;
        for (int u : node_set)
        {
            for (int v : adj[u])
            {
                cout << u << "  -->  " << v << endl;
            }
        }
    }

    // 树形字符图，缩进展示每一条边
    void print_graph_with_edge()
    {
        cout << "\n=== 字符图形化展示（每条边清晰可见）===\n";
        unordered_set<int> has_in;
        for (int u : node_set)
        {
            for (int v : adj[u])
                has_in.insert(v);
        }

        std::vector<int> start_nodes;
        for (int nd : node_set)
        {
            if (!has_in.count(nd))
                start_nodes.push_back(nd);
        }

        function<void(int, int)> dfs_draw = [&](int cur, int depth)
        {
            for (int i = 0; i < depth; ++i)
                cout << "    ";

            cout << "[" << cur << "]";
            if (!adj[cur].empty())
                cout << "  出边:";
            cout << endl;

            for (int nxt : adj[cur])
            {
                for (int i = 0; i < depth + 1; ++i)
                    cout << "    ";
                cout << "└──> 边: " << cur << " --> " << nxt << endl;
                dfs_draw(nxt, depth + 2);
            }
        };

        for (int s : start_nodes)
        {
            dfs_draw(s, 0);
        }
    }

    // 导出dot文件，Graphviz一键生成图片（最标准可视化）
    void export_dot_file(const string& name = "graph.dot")
    {
        ofstream out(name);
        out << "digraph G {\n";
        for (int u : node_set)
        {
            for (int v : adj[u])
            {
                out << "    " << u << " -> " << v << ";\n";
            }
        }
        out << "}\n";
        out.close();
        cout << "\n已生成 graph.dot\n使用命令：dot -Tpng graph.dot -o graph.png 生成图片\n";
    }
};

int main()
{
    DirectedGraph g;
    g.add_edge(0, 1);
    g.add_edge(0, 2);
    g.add_edge(1, 3);
    g.add_edge(2, 3);
    g.add_edge(3, 60);
    g.add_edge(60, 20);

    g.list_all();
    g.print_graph_with_edge();
    g.export_dot_file();
    return 0;
}
