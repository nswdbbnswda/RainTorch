#include <iostream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <set>
using namespace std;

class DirectedGraph {
public:
  DirectedGraph(int cap=50) {
    adj_.resize(cap);
    std::cout << "graph cap:" << cap << std::endl;
    indegree_.resize(cap);
  }

 ~DirectedGraph() = default;

  void add_node(int node) {
    nodes_.insert(node);
  }

  void add_edge(int u, int v) {
    auto max_id = std::max(u, v);
    add_node(u);
    add_node(v);
    if (max_id >= adj_.size()) {
      adj_.resize(max_id + 1); 
      indegree_.resize(max_id + 1);
    }
    adj_[u].push_back(v);
    indegree_[v]++;
  }

  void graph_node_count() {
    std::cout << "graph_node_count:" << nodes_.size() << std::endl;
  }
  
  void show_graph() {
    auto MakesubNodeStr = [](const std::vector<int>& subNodes) {
      std::string subNodeStr; 
      for (auto i = 0; i < subNodes.size(); ++i) {
        subNodeStr += std::to_string(subNodes[i]);
        if (i < subNodes.size() - 1) {
          subNodeStr += ",";
        }
      }
      return subNodeStr;
    };
    for (auto& node :  nodes_) {
      std::cout << "node: " << node << "----->" << MakesubNodeStr(adj_[node]) << std::endl;
    }
  }

  void dfs(int start) {
    std::unordered_set<int> visited;
    __dfs(start, visited);
  }

  void bfs(int start) {
    if (!nodes_.count(start)) {
      return;
    }
    std::queue<int> q;
    std::unordered_set<int> visited;
    q.push(start);
    visited.insert(start);
    while (!q.empty()) {
      auto cur = q.front();
      q.pop();
      std::cout << cur << " " << std::endl;
      for (const auto& neighbor : adj_[cur]) {
        if (!visited.count(neighbor)) {
           visited.insert(neighbor);
           q.push(neighbor);
        }
      }
    }
  }

  int get_indegree(int x) {
    return indegree_[x];
  }

  int get_outdegree(int x) {
    return adj_[x].size();
  }
  
  std::vector<int> kahn_topo() {
    std::queue<int> q;
    std::vector<int> topo_seq;
    std::vector<int> temp_indegree = indegree_;
    for (const auto& node : nodes_) {
      if (temp_indegree[node] == 0) {
        q.push(node);
      }
    }
    while (!q.empty()) {
      auto node = q.front();
      q.pop();
      topo_seq.push_back(node);
      auto& edges = adj_[node];
      for (const auto& edge : edges) {
        temp_indegree[edge]--;
        if (temp_indegree[edge] == 0) {
          q.push(edge);
        }
      }
    }
    if (topo_seq.size() != nodes_.size()) {
      return {};
    }
    return topo_seq;
  }
private:
  void __dfs(int cur, std::unordered_set<int>& visited) {
    visited.insert(cur);    
    std::cout << cur << std::endl;
    auto& edges = adj_[cur];
    for (const auto& neighbor : edges) {
      if (!visited.count(neighbor)) {
        __dfs(neighbor, visited);
      }
    }
  }
private:
  std::vector<std::vector<int>> adj_;
  std::set<int> nodes_;
  std::vector<int> indegree_;
};



int main() {
  DirectedGraph directed_graph;
  directed_graph.add_edge(1, 2);
  directed_graph.add_edge(1, 5);
  directed_graph.add_edge(2, 3);
  directed_graph.graph_node_count();
  directed_graph.show_graph();
  directed_graph.dfs(1);
  //std::cout << directed_graph.get_outdegree(1) << std::endl;
  //std::cout << directed_graph.get_indegree(2) << std::endl;
  auto topo_list = directed_graph.kahn_topo();
  std::cout << "topo sort result" << std::endl;
  for (auto &node : topo_list) {
    std::cout << node << " "; 
  }
  std::cout << std::endl;
  return 0;
}
