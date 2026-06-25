# Intel® Order Accuracy Reference Package

<!--hide_directive
<div class="component_card_widget">
  <a class="icon_github" href="https://github.com/intel-retail/order-accuracy/tree/main">
     GitHub
  </a>
  <a class="icon_document" href="https://github.com/intel-retail/order-accuracy/blob/main/README.md">
     Readme
  </a>
</div>
hide_directive-->

The Order Accuracy Pipeline System is an open-source reference implementation for building and deploying video analytics pipelines for retail order accuracy in Quick Service Restaurant (QSR) and Restaurant Dining use cases. It uses Intel® hardware and software, GStreamer, and OpenVINO™ to enable scalable, real-time object detection and classification at the edge. The platform automatically detects items in food trays, bags, or containers, compares them against expected order data, and identifies discrepancies before orders reach customers.

<!--hide_directive
::::{grid} 1 2 2 2
:::{grid-item-card} Dine-In
:class-card: homepage-card-container-big
:link: ./dine-in/index.html

Image-based order validation for restaurant dining applications.
:::
:::{grid-item-card} Take-Away
:class-card: homepage-card-container-big
:link: ./take-away/index.html

Real-time video stream validation for drive-through and counter service.
:::
::::
hide_directive-->

## Platform Applications

The platform provides two specialized applications optimized for different restaurant scenarios:

| Application                                | Use Case                            | Input Type           |
| ------------------------------------------ | ----------------------------------- | -------------------- |
| **[Dine-In](#dine-in-order-accuracy)**     | Restaurant table service validation | Static images        |
| **[Take-Away](#take-away-order-accuracy)** | Drive-through and counter service   | Video streams (RTSP) |

### Dine-In Order Accuracy

**Image-based order validation for restaurant dining applications**

Optimized for validating food trays at serving stations before delivery to tables. Uses single image capture and VLM analysis for fast, accurate item detection.

#### Key Features

- Single image capture and analysis
- Food tray/plate item detection
- REST API for POS integration
- Gradio web interface for manual validation
- Hybrid semantic matching
- Zero-training deployment with pre-trained Qwen2.5-VL-7B model

#### Use Case

In a full-service restaurant:

1. Kitchen prepares a dish for Table 12
2. Expo staff places the plate in the validation station
3. Staff triggers validation via Gradio UI or API
4. System analyzes plate contents using VLM
5. System compares detected items against the order manifest
6. Staff receives immediate feedback on order accuracy

### Take-Away Order Accuracy

**Real-time video stream validation for drive-through and counter service**

Optimized for high-throughput drive-through environments with multiple camera stations. Processes RTSP video streams in parallel using intelligent frame selection and VLM batching.

#### Key Features

- Real-time RTSP video stream processing
- Multi-station parallel processing (up to 8 workers)
- GStreamer-based video pipeline
- YOLO-powered frame selection
- VLM request batching for throughput
- Circuit breaker and auto-recovery

#### Service Modes

| Mode         | Description                 | Use Case              |
| ------------ | --------------------------- | --------------------- |
| **Single**   | Single worker, Gradio UI    | Development, testing  |
| **Parallel** | Multi-worker, VLM scheduler | Production deployment |

## Choosing the Right Application

| Criteria             | Dine-In             | Take-Away              |
| -------------------- | ------------------- | ---------------------- |
| **Input Type**       | Static images       | Video streams (RTSP)   |
| **Throughput**       | Low-medium          | High (parallel)        |
| **Latency Priority** | Accuracy over speed | Speed and accuracy     |
| **Camera Setup**     | Fixed position      | Multi-station          |
| **Typical Use**      | Table service       | Drive-through, counter |
| **Processing**       | Single request      | Batch processing       |

### Recommendation

- **Choose Dine-In** if you need to validate orders from captured images at serving stations
- **Choose Take-Away** if you need real-time validation from continuous video streams

---

## Shared Platform Components

### VLM Backend (OVMS)

Both applications use OpenVINO Model Server with Qwen2.5-VL for vision-language inference.

### Semantic Comparison Service

AI-powered semantic matching microservice for intelligent item comparison:

- **Matching Strategies**: Exact → Semantic → Hybrid
- **Example**: Matches "green apple" ↔ "apple" using semantic reasoning
- **Fallback**: Automatic fallback to local matching if service unavailable

## Next Steps

- [Get Started Guide](./oa-get-started.md) - **Quick Start (25 minutes)**
  - Set up your development environment
  - Run your first order validation demo
  - Understand the platform workflow

- [Advanced Settings](./get-started/advanced.md) - **Advanced Configuration (45-90 minutes)**
  - Configure custom settings and workloads
  - Tune performance parameters
  - Set up benchmarking

- [How It Works](./oa-how-it-works.md) - **System Design**
  - Understand system components
  - Review data flow diagrams
  - Learn about service interactions

- [Benchmarking](./oa-benchmarking.md) - **Benchmark & Optimize**
  - Compare CPU/GPU performance
  - Run stream density tests
  - Optimize for your hardware

<!--hide_directive
:::{toctree}
:hidden:

./oa-get-started
How It Works <./oa-how-it-works>
Dine-In Order Accuracy <./dine-in/index>
Take-Away Order Accuracy <./take-away/index>
Benchmarking <./oa-benchmarking>

:::
hide_directive-->
