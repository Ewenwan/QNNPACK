# QNNPACK
QNNPACK (Quantized Neural Networks PACKage) is a mobile-optimized library for low-precision high-performance neural network inference. QNNPACK provides implementation of convolutional, deconvolutional, and fully connected neural network operators on quantized 8-bit tensors.

QNNPACK is not intended to be directly used by machine learning researchers; instead it provides low-level performance primitives for high-level deep learning frameworks. As of today, QNNPACK is integrated in [PyTorch 1.0](https://github.com/pytorch/pytorch) with Caffe2 graph representation.

## Building

QNNPACK provides standard CMake-based build scripts.

### Native compilation

Users are recommended to use `scripts/build-local.sh` script to build QNNPACK for the host machine. 

### Cross-compilation for Android

To cross-compile for Android, set `$ANDROID_NDK` environment variable (where `$ANDROID_NDK` is the path to Android NDK directorory, e.g. `/opt/android-ndk-r15c`) and use one of the scripts from the table below:

| ABI         | Build script                     | Restrictions               |
| ----------- | ---------------------------------| -------------------------- |
| armeabi-v7a | `scripts/build-android-armv7.sh` | Requires CPU with ARM NEON |
| arm64-v8a   | `scripts/build-android-arm64.sh` |                            |
| x86         | `scripts/build-android-x86.sh`   |                            |

Notes:
- On **armeabi-v7a** `qnnp_initialize` will fail with `qnnp_status_unsupported_hardware` if the mobile CPU does not support ARM NEON. Don't set `-DANDROID_ARM_NEON=1` for QNNPACK compilation as it can make `qnnp_initialize` crash on CPUs without ARM NEON.

### Cross-compilation for iOS

To cross-compile for iOS, use [ios-cmake](https://github.com/leetal/ios-cmake) to generate Xcode project files.

## Acknowledgements

QNNPACK is developed by Marat Dukhan, Yiming Wu, Hao Lu, and Bert Maher. We thank Andrew Tulloch and Yangqing Jia for advice during the development of QNNPACK.

## License

QNNPACK is BSD licensed, as found in the [`LICENSE`](LICENSE) file.


## 简介
    QNNPACK 的全称是 Quantized Neural Network PACKage（量化神经网络包），
    是 Facebook 应用的一部分，已经被部署到全球数十亿台移动设备中。这个新库可以执行高级计算机视觉任务，
    如在手机上实时运行 Mask R-CNN 和 DensePose 或在性能受限的移动设备中用 100ms 以内的时间实施图像分类。

    Facebook 开源 QNNPACK，为优化推理提供全方位的支持，作为构建 PyTorch 1.0 平台的一部分。
    QNNPACK 借助 Caffe2 模型表征即刻可用，Facebook 正在开发实用程序，将 PyTorch 的 Python 前端模型导出到图表征中。
    他们还在其他平台上优化这些运算，而不仅限于移动设备。

    由于移动设备的计算力仅仅是数据中心服务器的十分之一到千分之一，运行当前最佳人工智能应用需要作出一些调整，
    压缩来自硬件的所有可用性能。QNNPACK 通过提供量化张量上的卷积、解卷积及全连接运算高性能实现来做到这一点。
    在 QNNPACK 之前，几个常见的神经网络基元（分组卷积、扩张卷积）缺乏良好的开源实现；
    因此，ResNeXt、CondenseNet 和 ShuffleNet 等颇有前景的研究模型没有得到充分利用。

## 移动设备前沿 AI 技术新优化

    两年前，Facebook 开始在手机上部署神经网络，多数计算机视觉架构随着大型内核被部署到卷积运算中。
    这些运算因计算强度高而饱受诟病：直接实现涉及每个加载元素的许多乘-加运算。
    Caffe2Go 使用的是一种叫做 NNPACK 的内核库，该库实现基于 Winograd 变换或快速傅立叶变换的渐近快速卷积算法，
    以减少卷积计算中的乘-加运算。例如，3×3 卷积比 1×1 卷积运算慢两倍，但使用直接算法要慢 9 倍。

    计算机视觉领域发展迅猛，然而，这种新的神经网络架构使用的是几种无法从快速卷积算法中获益的卷积，
    即 1×1 卷积、分组卷积、转置卷积、空洞卷积和深度卷积。这些类型的卷积计算强度相对较低，
    因此可以通过利用低精度计算从内存降低的带宽中受益。

    用于计算机视觉的神经网络将多数推理时间用在卷积和全连接算子中。
    这些算子与矩阵相乘紧密相关：全连接算子和 1×1 卷积直接映射到矩阵相乘，
    具有较大内核的卷积可以分解成一种名为 im2col 的内存布局转换和矩阵相乘的组合。
    因此，卷积神经网络中的有效推理问题很大程度上可以看做矩阵乘法的有效实现问题——在线性代数库中也称为 GEMM。

## 实现矩阵相乘

    不直接在科学计算或者深度学习软件上工作的软件工程师可能不熟悉库是如何实现矩阵相乘的，
    所以在详细介绍 QNNPACK 之前，会有一个总体介绍。

    在以下示例中，A 是输入，B 是权重，C 是输出。
    在推理过程中，B 从不变化，也因此不需要消耗时间就能迁移到任何方便的存储配置中。

## 神经网络中的优化及 QNNPACK 如何提高效率
    PyTorch 及其它深度学习框架在训练期间通常利用浮点数来表示权重和神经网络的神经元。
    模型训练完成之后，浮点数及运算就会显得过分：许多类型的模型可以在调整后使用推理用的低精度整数运算，
    不会出现明显的准确率损失。低精度整数表征在单精度、甚至是半精度浮点上提供一些益处：
    内存占用减小 2/1 或 3/4，有助于将神经网络模型保存在移动处理器的小缓存中；
    提高内存带宽受限的运算性能；
    提高能源利用率；
    在许多类型的硬件上提高计算吞吐量。

    QNNPACK 使用与安卓神经网络 API 兼容的线性量化方案。
    它假设量化值 q[i] 表示为 8 位无符号整数，
    并且它们与实值表示 r[i] 相关，公式如下：

    r[i] = scale * (q[i] – zero_point)

    公式中的 scale 是一个正浮点数，zero_point 是一个无符号的 8 位整数，就像 q[i] 一样。

    尽管 QNNPACK 像其它 BLAS 库一样利用 PDOT 微内核，但它对具有 8 位元素的量化张量和
    移动 AI 用例的关注为性能优化带来了截然不同的视角。
    多数 BLAS 库针对的是矩阵高达数千个双精度浮点元素的科学计算用例，
    但 QNNPACK 的输入矩阵来自低精度、移动专用的计算机视觉模型，并且具有非常不同的维度。
    在 1×1 卷积中，K 是输入通道的数量，N 是输出通道的数量，M 是图像中像素的数量。
    在实用移动优化网络中，K 和 N 不超过 1024，取值范围通常在 32-256 之间。

    由于移动架构的局限，MR 和 NR 不超过 8。因此即使是在有 1024 个通道的最大模型中，
    整个内存块在 PDOT 微内核中的读取速度也只能达到 16KB，即使在超低端移动内核上也能适用于一级缓存。
    这标志着 QNNPACK 和其他 GEMM 实现之间的一个重要区别：虽然其它库重新打包 A 和 B 矩阵以更好地利用缓存层次结构，
    希望在大量计算中分摊打包开销，但 QNNPACK 针对 A 和 B 的面板适用于一级缓存的情况进行了优化。
    因此，它的目的是删除所有计算非必需的内存转换。
    
    在量化矩阵-矩阵乘法中，8 位整数的乘积通常会被累加至 32 位的中间结果中，随后重新量化以产生 8 位的输出。
    常规的实现会对大矩阵尺寸进行优化——有时 K 太大无法将 A 和 B 的面板转入缓存中。
    为了有效利用缓存层次结构，传统的 GEMM 实现将 A 和 B 的面板沿 K 维分割成固定大小的子面板，
    从而每个面板都适应 L1 缓存，随后为每个子面板调用微内核。
    这一缓存优化需要 PDOT 为内核输出 32 位中间结果，最终将它们相加并重新量化为 8 位整数。

    由于 ONNPACK 对于面板 A 和 B 总是适应 L1 缓存的移动神经网络进行了优化，
    因此它在调用微内核时处理整个 A 和 B 的面板。而由于无需在微内核之外积累 32 位的中间结果，
    QNNPACK 会将 32 位的中间结果整合进微内核中并写出 8 位值，这节省了内存带宽和缓存占用。
    
    
## QNNPACK 和深度卷积

    分组卷积（grouped convolution）将输入和输出通道分割成多组，然后对每个组进行分别处理。
    在有限条件下，当组数等于通道数时，该卷积就是深度卷积，常用于当前的神经网络架构中。
    深度卷积对每个通道分别执行空间滤波，展示了与正常卷积非常不同的计算模式。
    因此，通常要向深度卷积提供单独实现，QNNPACK 包括一个高度优化版本 3×3 深度卷积。

    深度卷积的传统实现是每次都在卷积核元素上迭代，然后将一个卷积核行和一个输入行的结果累加到输出行。
    对于一个 3×3 的深度卷积，此类实现将把每个输出行更新 9 次。在 QNNPACK 中，
    研究者计算所有 3×3 卷积核行和 3×3 输入行的结果，一次性累加到输出行，然后再处理下个输出行。

    QNNPACK 实现高性能的关键因素在于完美利用通用暂存器（GPR）来展开卷积核元素上的循环，
    同时避免在 hot loop 中重新加载地址寄存器。32-bit ARM 架构将实现限制在 14 个 GPR。
    在 3×3 深度卷积中，需要读取 9 个输入行和 9 个卷积核行。这意味着如果想完全展开循环必须存储 18 个地址。
    然而，实践中推断时卷积核不会发生变化。因此 Facebook 研究者使用之前在 CxKHxKW 中的滤波器，
    将它们封装进 [C/8]xKWxKHx8，这样就可以仅使用具备地址增量（address increment）的一个 GPR 访问所有滤波器。
    （研究者使用数字 8 的原因在于，在一个命令中加载 8 个元素然后减去零，在 128-bit NEON 暂存器中生成 8 个 16-bit 值。）
    然后使用 9 个输入行指针，指针将滤波器重新装进 10 个 GPR，完全展开滤波器元素上的循环。
    64-bit ARM 架构相比 32-bit 架构，GPR 的数量翻了一倍。
    QNNPACK 利用额外的 ARM64 GPR，一次性存储 3×5 输入行的指针，并计算 3 个输出行。

## QNNPACK 的性能优势

    测试结果显示出 QNNPACK 在端到端基准上的性能优势。
    在量化当前最优 MobileNetV2 架构上，基于 QNNPACK 的 Caffe2 算子的速度大约是 TensorFlow Lite 速度的 2 倍，
    在多种手机上都是如此。除了 QNNPACK 之外，Facebook 还开源了 Caffe2 quantized MobileNet v2 模型，
    其 top-1 准确率比相应的 TensorFlow 模型高出 1.3%。

    Caffe2 quantized MobileNet v2 模型开源地址：https://github.com/caffe2/models/tree/master/mobilenet_v2_quantized

