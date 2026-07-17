#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <unordered_set>

struct Node {
    double val;
    double grad;
    std::vector<std::shared_ptr<Node>> parents;
    std::function<void()> grad_fn;
    bool requires_grad;
    bool is_leaf;
    Node(double v, bool requires_grad=true, bool is_leaf=false) : val(v), grad(0.0), requires_grad(requires_grad), is_leaf(is_leaf) {}
};

using Var = std::shared_ptr<Node>;


Var create(double val, bool requires_grad=true, bool is_leaf=false) {
    return std::make_shared<Node>(val, requires_grad, is_leaf);
}

struct SGD {
  double lr;
  std::vector<Var> params;
  SGD(double lr, const std::vector<Var>& params) : lr(lr), params(params) {}
  void step() {
    for (auto& v : params) {
      v->val = v->val - lr * v->grad;
    }
  }

  void zero_grad() {
    for (auto& v : params) {
      if (v->is_leaf) {
        v->grad = 0;
      }
    }
  }
};

void build_topo(Var node, std::vector<Var>& topo, std::unordered_set<Node*>& visited) {
    if (!node) return;
    Node* raw = node.get();
    if (visited.count(raw)) return;
    visited.insert(raw);

    for (auto& p : node->parents) {
        build_topo(p, topo, visited);
    }
    if (node->requires_grad) {
      topo.push_back(node);
    }
}

void backward(Var out, std::vector<Var>& topo) {
    if (!out) return;
    out->grad = 1.0;

    std::unordered_set<Node*> visited;
    build_topo(out, topo, visited);

    std::reverse(topo.begin(), topo.end());
    for (auto& n : topo) {
      if (n->grad_fn) {
        n->grad_fn();
      }
    }

    // 新增：非叶子节点梯度直接置0，节省内存
    for (auto& n : topo) {
        if (!n->is_leaf) {
            n->grad = 0.0;
        }
    }
}

// 四则算子
Var operator+(Var a, Var b) {
    bool need_grad = a->requires_grad || b->requires_grad;
    Var res = create(a->val + b->val, need_grad);
    res->parents = {a, b};
    if (need_grad) {
      res->grad_fn = [a, b, res]() {
          a->grad += res->grad;
          b->grad += res->grad;
      };
    }
    return res;
}

Var operator*(Var a, Var b) {
    bool need_grad = a->requires_grad || b->requires_grad;
    Var res = create(a->val * b->val, need_grad);
    res->parents = {a, b};
    if (need_grad) {
      res->grad_fn = [a, b, res]() {
        a->grad += res->grad * b->val;
        b->grad += res->grad * a->val;
      };
    }
    return res;
}

Var operator-(Var a, Var b) {
    bool need_grad = a->requires_grad || b->requires_grad; 
    Var res = create(a->val - b->val, need_grad);
    res->parents = {a, b};
    if (need_grad) {
      res->grad_fn = [a, b, res](){
          a->grad += res->grad;
          b->grad -= res->grad;
      };
    }
    return res;
}

Var operator/(Var a, Var b) {
    bool need_grad = a->requires_grad || b->requires_grad;
    Var res = create(a->val / b->val, need_grad);
    res->parents = {a, b};
    if (need_grad) {
        res->grad_fn = [a, b, res](){
            a->grad += res->grad / b->val;
            b->grad -= res->grad * a->val / (b->val * b->val);
        };
    }
    return res;
}

// 一元负号 -Var
Var operator-(Var x) {
    bool need_grad = x->requires_grad;
    Var res = create(-x->val, need_grad);
    res->parents = {x};
    if (need_grad) {
        res->grad_fn = [x, res](){
            x->grad -= res->grad;
        };
    }
    return res;
}

// ReLU
Var relu(Var x) {
    bool need_grad = x->requires_grad;
    double out = x->val > 0 ? x->val : 0.0;
    Var res = create(out, need_grad);
    res->parents = {x};
    if (need_grad) {
        res->grad_fn = [x, res](){
            if (x->val > 0) {
                x->grad += res->grad;
            }
        };
    }
    return res;
}

// 求和
Var sum(Var x) {
    // 当前标量版本sum就是自身，后续向量版再改写
    return x;
}

// 平方，用于损失函数
Var square(Var x) {
    bool need_grad = x->requires_grad;
    Var res = create(x->val * x->val, need_grad);
    res->parents = {x};
    if (need_grad) {
      res->grad_fn = [x, res](){
        x->grad += res->grad * 2 * x->val;
      };
    }
    return res;
}

struct Module {
  virtual ~Module() = default;

  // 前向传播算子, 子类必须实现
  virtual Var forward(Var x) = 0;

  virtual std::vector<Var> parameters() {
    std::vector<Var> res;
    for (auto& m : subs) {
      auto p = m->parameters();
      res.insert(res.end(), p.begin(), p.end());
    }
    return res;
  }

  void register_submodule(std::shared_ptr<Module> sub) {
    subs.push_back(sub);
  }

private:
  std::vector<std::shared_ptr<Module>> subs;
};

struct Linear : Module {
  Var w;
  Var b;

  Linear(int in_dim, int out_dim) {
    // 权重偏置默认开启梯度, 可训练
    w = create(0, true, true);
    b = create(0, true, true);
  }

  virtual Var forward(Var x) override {
    return w * x + b;
  }

  virtual std::vector<Var> parameters() override {
    return {w, b};
  }
};

struct TwoLayerNet : Module {
    std::shared_ptr<Linear> fc1, fc2;
    TwoLayerNet() {
        fc1 = std::make_shared<Linear>(1, 4);
        fc2 = std::make_shared<Linear>(4, 1);
        register_submodule(fc1);
        register_submodule(fc2);
    }

    Var forward(Var x) override {
        x = fc1->forward(x);
        x = relu(x);
        x = fc2->forward(x);
        return x;
    }
};


int main() {
    // 构造数据集 y = 3*x + 4
    std::vector<double> xs = {1, 2, 3, 4, 5};
    std::vector<double> ys = {7, 10, 13, 16, 19};

    Linear model(1, 1);
    std::vector<Var> params = model.parameters();
    double lr = 0.01;
    SGD sgd(lr, params);

    int epochs = 1000;

    for (int epoch = 0; epoch < epochs; epoch++) {
        sgd.zero_grad();
        Var total_loss = create(0.0, false);
        std::vector<Var> topo_nodes;
        for (int i = 0; i < xs.size(); i++) {
            double xi = xs[i];
            double yi_true = ys[i];

            Var x = create(xi, false);
            Var y_true = create(yi_true, false);

            Var y_pred = model.forward(x);
            Var loss = square(y_pred - y_true);

            total_loss = total_loss + loss;
        }

        backward(total_loss, topo_nodes);
        sgd.step();

        if (epoch % 50 == 0) {
            std::cout << "Epoch " << epoch
                      << " | w=" << model.w->val
                      << " | b=" << model.b->val
                      << " | total_loss=" << total_loss->val << "\n";
        }
    }

    std::cout << "\n训练完成！\n";
    std::cout << "拟合得到 w ≈ " << model.w->val << "\n";
    std::cout << "拟合得到 b ≈ " << model.b->val << "\n";
    std::cout << "理论真值 w=3, b=4\n";
    return 0;
}
