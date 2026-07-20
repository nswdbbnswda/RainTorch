#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <unordered_set>
#include <stdexcept>
#include <functional>
#include <random>

// 随机数初始化权重
std::mt19937 rng(42);
std::uniform_real_distribution<double> dist(-0.1, 0.1);

// ===================== 张量工具函数 =====================
inline int numel(const std::vector<int>& shape) {
    int cnt = 1;
    for (int d : shape) cnt *= d;
    return cnt;
}

std::vector<int> broadcast_shape(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<int> res;
    int i = (int)a.size() - 1, j = (int)b.size() - 1;
    while (i >= 0 || j >= 0) {
        int da = (i >= 0) ? a[i] : 1;
        int db = (j >= 0) ? b[j] : 1;
        if (da != 1 && db != 1 && da != db) {
            throw std::runtime_error("broadcast failed: shape mismatch");
        }
        res.push_back(std::max(da, db));
        i--; j--;
    }
    std::reverse(res.begin(), res.end());
    return res;
}

int coord2offset(const std::vector<int>& coord, const std::vector<int>& shape) {
    int off = 0;
    int stride = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        off += coord[i] * stride;
        stride *= shape[i];
    }
    return off;
}

int broadcast_offset(const std::vector<int>& out_coord,
                     const std::vector<int>& out_shape,
                     const std::vector<int>& in_shape) {
    std::vector<int> in_coord(in_shape.size(), 0);
    int diff = (int)out_shape.size() - (int)in_shape.size();
    for (int i = 0; i < (int)in_shape.size(); ++i) {
        int out_dim_idx = diff + i;
        int out_c = out_coord[out_dim_idx];
        int in_d = in_shape[i];
        in_coord[i] = (in_d == 1) ? 0 : out_c;
    }
    return coord2offset(in_coord, in_shape);
}

// ===================== Node 张量节点 =====================
struct Node {
    std::vector<double> data;
    std::vector<double> grad;
    std::vector<int> shape;
    std::vector<std::shared_ptr<Node>> parents;
    std::function<void()> grad_fn;
    bool requires_grad;
    bool is_leaf;

    Node(std::vector<double> d, std::vector<int> shp, bool req_grad = true, bool leaf = false)
        : data(std::move(d)), shape(std::move(shp)), requires_grad(req_grad), is_leaf(leaf)
    {
        int n = numel(shape);
        grad.assign(n, 0.0);
    }
};
using Var = std::shared_ptr<Node>;

// 创建张量接口
Var create(std::vector<double> data, std::vector<int> shape, bool requires_grad = true, bool is_leaf = false) {
    return std::make_shared<Node>(std::move(data), std::move(shape), requires_grad, is_leaf);
}
Var create(double val, bool requires_grad = true, bool is_leaf = false) {
    return create({val}, {1}, requires_grad, is_leaf);
}

// ===================== 反向传播拓扑排序 =====================
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

void backward(Var out) {
    std::vector<Var> topo;
    std::unordered_set<Node*> visited;
    build_topo(out, topo, visited);
    std::fill(out->grad.begin(), out->grad.end(), 1.0);
    std::reverse(topo.begin(), topo.end());
    for (auto& n : topo) {
        if (n->grad_fn) n->grad_fn();
    }
    // 清空非叶子节点梯度节省内存
    for (auto& n : topo) {
        if (!n->is_leaf) {
            std::fill(n->grad.begin(), n->grad.end(), 0.0);
        }
    }
}

// ===================== 广播四则运算符 =====================
Var operator+(Var a, Var b) {
    bool need_grad = a->requires_grad || b->requires_grad;
    std::vector<int> out_shape = broadcast_shape(a->shape, b->shape);
    int out_n = numel(out_shape);
    std::vector<double> out_data(out_n, 0.0);
    std::vector<int> coord(out_shape.size(), 0);

    for (int idx = 0; idx < out_n; ++idx) {
        int tmp = idx;
        for (int d = (int)out_shape.size() - 1; d >= 0; --d) {
            coord[d] = tmp % out_shape[d];
            tmp /= out_shape[d];
        }
        int off_a = broadcast_offset(coord, out_shape, a->shape);
        int off_b = broadcast_offset(coord, out_shape, b->shape);
        out_data[idx] = a->data[off_a] + b->data[off_b];
    }
    Var res = create(out_data, out_shape, need_grad);
    res->parents = {a, b};
    if (need_grad) {
        res->grad_fn = [a, b, res, out_shape]() {
            int n = numel(out_shape);
            std::vector<int> coord(out_shape.size(), 0);
            for (int idx = 0; idx < n; ++idx) {
                int tmp = idx;
                for (int d = (int)out_shape.size() - 1; d >= 0; --d) {
                    coord[d] = tmp % out_shape[d];
                    tmp /= out_shape[d];
                }
                double dout = res->grad[idx];
                int off_a = broadcast_offset(coord, out_shape, a->shape);
                int off_b = broadcast_offset(coord, out_shape, b->shape);
                a->grad[off_a] += dout;
                b->grad[off_b] += dout;
            }
        };
    }
    return res;
}

Var operator*(Var a, Var b) {
    bool need_grad = a->requires_grad || b->requires_grad;
    std::vector<int> out_shape = broadcast_shape(a->shape, b->shape);
    int out_n = numel(out_shape);
    std::vector<double> out_data(out_n, 0.0);
    std::vector<int> coord(out_shape.size(), 0);

    for (int idx = 0; idx < out_n; ++idx) {
        int tmp = idx;
        for (int d = (int)out_shape.size() - 1; d >= 0; --d) {
            coord[d] = tmp % out_shape[d];
            tmp /= out_shape[d];
        }
        int off_a = broadcast_offset(coord, out_shape, a->shape);
        int off_b = broadcast_offset(coord, out_shape, b->shape);
        out_data[idx] = a->data[off_a] * b->data[off_b];
    }
    Var res = create(out_data, out_shape, need_grad);
    res->parents = {a, b};
    if (need_grad) {
        res->grad_fn = [a, b, res, out_shape]() {
            int n = numel(out_shape);
            std::vector<int> coord(out_shape.size(), 0);
            for (int idx = 0; idx < n; ++idx) {
                int tmp = idx;
                for (int d = (int)out_shape.size() - 1; d >= 0; --d) {
                    coord[d] = tmp % out_shape[d];
                    tmp /= out_shape[d];
                }
                double dout = res->grad[idx];
                int off_a = broadcast_offset(coord, out_shape, a->shape);
                int off_b = broadcast_offset(coord, out_shape, b->shape);
                a->grad[off_a] += dout * b->data[off_b];
                b->grad[off_b] += dout * a->data[off_a];
            }
        };
    }
    return res;
}

Var operator-(Var a, Var b) {
    bool need_grad = a->requires_grad || b->requires_grad;
    std::vector<int> out_shape = broadcast_shape(a->shape, b->shape);
    int out_n = numel(out_shape);
    std::vector<double> out_data(out_n, 0.0);
    std::vector<int> coord(out_shape.size(), 0);

    for (int idx = 0; idx < out_n; ++idx) {
        int tmp = idx;
        for (int d = (int)out_shape.size() - 1; d >= 0; --d) {
            coord[d] = tmp % out_shape[d];
            tmp /= out_shape[d];
        }
        int off_a = broadcast_offset(coord, out_shape, a->shape);
        int off_b = broadcast_offset(coord, out_shape, b->shape);
        out_data[idx] = a->data[off_a] - b->data[off_b];
    }
    Var res = create(out_data, out_shape, need_grad);
    res->parents = {a, b};
    if (need_grad) {
        res->grad_fn = [a, b, res, out_shape]() {
            int n = numel(out_shape);
            std::vector<int> coord(out_shape.size(), 0);
            for (int idx = 0; idx < n; ++idx) {
                int tmp = idx;
                for (int d = (int)out_shape.size() - 1; d >= 0; --d) {
                    coord[d] = tmp % out_shape[d];
                    tmp /= out_shape[d];
                }
                double dout = res->grad[idx];
                int off_a = broadcast_offset(coord, out_shape, a->shape);
                int off_b = broadcast_offset(coord, out_shape, b->shape);
                a->grad[off_a] += dout;
                b->grad[off_b] -= dout;
            }
        };
    }
    return res;
}

Var operator/(Var a, Var b) {
    bool need_grad = a->requires_grad || b->requires_grad;
    std::vector<int> out_shape = broadcast_shape(a->shape, b->shape);
    int out_n = numel(out_shape);
    std::vector<double> out_data(out_n, 0.0);
    std::vector<int> coord(out_shape.size(), 0);

    for (int idx = 0; idx < out_n; ++idx) {
        int tmp = idx;
        for (int d = (int)out_shape.size() - 1; d >= 0; --d) {
            coord[d] = tmp % out_shape[d];
            tmp /= out_shape[d];
        }
        int off_a = broadcast_offset(coord, out_shape, a->shape);
        int off_b = broadcast_offset(coord, out_shape, b->shape);
        out_data[idx] = a->data[off_a] / b->data[off_b];
    }
    Var res = create(out_data, out_shape, need_grad);
    res->parents = {a, b};
    if (need_grad) {
        res->grad_fn = [a, b, res, out_shape]() {
            int n = numel(out_shape);
            std::vector<int> coord(out_shape.size(), 0);
            for (int idx = 0; idx < n; ++idx) {
                int tmp = idx;
                for (int d = (int)out_shape.size() - 1; d >= 0; --d) {
                    coord[d] = tmp % out_shape[d];
                    tmp /= out_shape[d];
                }
                double dout = res->grad[idx];
                int off_a = broadcast_offset(coord, out_shape, a->shape);
                int off_b = broadcast_offset(coord, out_shape, b->shape);
                double va = a->data[off_a];
                double vb = b->data[off_b];
                a->grad[off_a] += dout / vb;
                b->grad[off_b] -= dout * va / (vb * vb);
            }
        };
    }
    return res;
}

Var operator-(Var x) {
    bool need_grad = x->requires_grad;
    int n = numel(x->shape);
    std::vector<double> out_data(n);
    for (int i = 0; i < n; ++i) out_data[i] = -x->data[i];
    Var res = create(out_data, x->shape, need_grad);
    res->parents = {x};
    if (need_grad) {
        res->grad_fn = [x, res]() {
            int n = numel(res->shape);
            for (int i = 0; i < n; ++i) {
                x->grad[i] -= res->grad[i];
            }
        };
    }
    return res;
}

// ===================== 激活与损失算子 =====================
Var relu(Var x) {
    bool need_grad = x->requires_grad;
    int n = numel(x->shape);
    std::vector<double> out_data(n);
    for (int i = 0; i < n; ++i) {
        out_data[i] = x->data[i] > 0 ? x->data[i] : 0.0;
    }
    Var res = create(out_data, x->shape, need_grad);
    res->parents = {x};
    if (need_grad) {
        res->grad_fn = [x, res]() {
            int n = numel(res->shape);
            for (int i = 0; i < n; ++i) {
                if (x->data[i] > 0) {
                    x->grad[i] += res->grad[i];
                }
            }
        };
    }
    return res;
}

Var square(Var x) {
    bool need_grad = x->requires_grad;
    int n = numel(x->shape);
    std::vector<double> out_data(n);
    for (int i = 0; i < n; ++i) {
        out_data[i] = x->data[i] * x->data[i];
    }
    Var res = create(out_data, x->shape, need_grad);
    res->parents = {x};
    if (need_grad) {
        res->grad_fn = [x, res]() {
            int n = numel(res->shape);
            for (int i = 0; i < n; ++i) {
                x->grad[i] += res->grad[i] * 2 * x->data[i];
            }
        };
    }
    return res;
}

Var sum(Var x) {
    bool need_grad = x->requires_grad;
    int n = numel(x->shape);
    double s = 0.0;
    for (double v : x->data) s += v;
    Var res = create(s, need_grad);
    res->parents = {x};
    if (need_grad) {
        res->grad_fn = [x, res]() {
            double dout = res->grad[0];
            int n = numel(x->shape);
            for (int i = 0; i < n; ++i) {
                x->grad[i] += dout;
            }
        };
    }
    return res;
}

// ===================== 矩阵转置 =====================
Var transpose(Var x) {
    if (x->shape.size() != 2) throw std::runtime_error("transpose only support 2D");
    int N = x->shape[0];
    int K = x->shape[1];
    std::vector<double> out_data(N * K, 0.0);
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            int src = n * K + k;
            int dst = k * N + n;
            out_data[dst] = x->data[src];
        }
    }
    Var res = create(out_data, {K, N}, x->requires_grad);
    res->parents = {x};
    if (x->requires_grad) {
        res->grad_fn = [x, res, N, K]() {
            for (int n = 0; n < N; n++) {
                for (int k = 0; k < K; k++) {
                    int src = k * N + n;
                    int dst = n * K + k;
                    x->grad[dst] += res->grad[src];
                }
            }
        };
    }
    return res;
}

// ===================== 修复版matmul：无递归，手写三重循环梯度 =====================
Var matmul(Var A, Var B) {
    if (A->shape.size() != 2 || B->shape.size() != 2)
        throw std::runtime_error("matmul only support 2D matrix");
    int N = A->shape[0];
    int K = A->shape[1];
    int K2 = B->shape[0];
    int M = B->shape[1];
    if (K != K2) throw std::runtime_error("matmul dim mismatch A.col != B.row");

    bool need_grad = A->requires_grad || B->requires_grad;
    std::vector<int> out_shape = {N, M};
    int out_n = N * M;
    std::vector<double> out_data(out_n, 0.0);

    // 前向
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            double s = 0.0;
            for (int k = 0; k < K; k++) {
                int offA = n * K + k;
                int offB = k * M + m;
                s += A->data[offA] * B->data[offB];
            }
            out_data[n * M + m] = s;
        }
    }

    Var res = create(out_data, out_shape, need_grad);
    res->parents = {A, B};
    if (need_grad) {
        res->grad_fn = [A, B, res, N, K, M]() {
            // dA = dout @ B.T
            for (int n = 0; n < N; n++) {
                for (int k = 0; k < K; k++) {
                    double ga = 0.0;
                    for (int m = 0; m < M; m++) {
                        int idx_d = n * M + m;
                        int idx_b = k * M + m;
                        ga += res->grad[idx_d] * B->data[idx_b];
                    }
                    int idx_a = n * K + k;
                    A->grad[idx_a] += ga;
                }
            }
            // dB = A.T @ dout
            for (int k = 0; k < K; k++) {
                for (int m = 0; m < M; m++) {
                    double gb = 0.0;
                    for (int n = 0; n < N; n++) {
                        int idx_d = n * M + m;
                        int idx_a = n * K + k;
                        gb += A->data[idx_a] * res->grad[idx_d];
                    }
                    int idx_b = k * M + m;
                    B->grad[idx_b] += gb;
                }
            }
        };
    }
    return res;
}

// ===================== SGD优化器 =====================
struct SGD {
    double lr;
    std::vector<Var> params;
    SGD(double lr, const std::vector<Var>& params) : lr(lr), params(params) {}

    void step() {
        for (auto& v : params) {
            int n = numel(v->shape);
            for (int i = 0; i < n; ++i) {
                v->data[i] -= lr * v->grad[i];
            }
        }
    }
    void zero_grad() {
        for (auto& v : params) {
            if (v->is_leaf) {
                std::fill(v->grad.begin(), v->grad.end(), 0.0);
            }
        }
    }
};

// ===================== Module 网络基类 =====================
struct Module {
    virtual ~Module() = default;
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

// 支持批量输入的Linear层，随机初始化权重
struct Linear : Module {
    Var w;
    Var b;
    int in_dim;
    int out_dim;

    Linear(int in_, int out_) : in_dim(in_), out_dim(out_) {
        std::vector<double> w_data(in_ * out_);
        for (auto& v : w_data) v = dist(rng);
        w = create(w_data, {out_dim, in_dim}, true, true);

        std::vector<double> b_data(out_dim, 0.0);
        b = create(b_data, {out_dim}, true, true);
    }

    Var forward(Var x) override {
        Var wx = matmul(x, transpose(w));
        return wx + b;
    }

    std::vector<Var> parameters() override {
        return {w, b};
    }
};

// 两层神经网络
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

// ===================== 训练入口 =====================
int main() {
    // 输入批量 shape [5, 1]
    std::vector<double> xs_data = {1, 2, 3, 4, 5};
    Var x_batch = create(xs_data, {5, 1}, false);
    std::vector<double> ys_data = {7, 10, 13, 16, 19};
    Var y_batch = create(ys_data, {5, 1}, false);

    Linear model(1, 1);
    auto params = model.parameters();
    SGD sgd(0.01, params); // 调高学习率

    int epochs = 2000;
    for (int e = 0; e < epochs; e++) {
        sgd.zero_grad();
        Var pred = model.forward(x_batch);
        Var loss_vec = square(pred - y_batch);
        Var loss = sum(loss_vec);
        backward(loss);
        sgd.step();

        if (e % 100 == 0) {
            double w0 = model.w->data[0];
            double b0 = model.b->data[0];
            std::cout << "Epoch " << e
                      << " w=" << w0
                      << " b=" << b0
                      << " loss=" << loss->data[0] << "\n";
        }
    }

    std::cout << "\n==== 训练完成 ====\n";
    std::cout << "拟合权重 w ≈ " << model.w->data[0] << "\n";
    std::cout << "拟合偏置 b ≈ " << model.b->data[0] << "\n";
    std::cout << "理论目标 w=3, b=4\n";
    return 0;
}
