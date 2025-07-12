#include "xfeat.hpp"
#include <iostream>

int main()
{
    using std::cout;
    using std::endl;

    matx::cudaExecutor exec{0};
    XFeat feat(exec);

    auto result = feat.keypointHead();
    exec.sync();

    matx::print_shape(result);

    return 0;
}