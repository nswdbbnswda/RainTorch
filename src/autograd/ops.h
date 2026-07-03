#pragma once
#include "autograd/node.h"

namespace raintorch {

NodePtr operator+(const NodePtr& a, const NodePtr& b);
NodePtr operator*(const NodePtr& a, const NodePtr& b);
NodePtr operator-(const NodePtr& a, const NodePtr& b);
NodePtr operator-(const NodePtr& a);
NodePtr pow2(const NodePtr& a);

}
