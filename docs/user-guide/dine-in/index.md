# Dine-In Order Accuracy

<!--hide_directive
<div class="component_card_widget">
  <a class="icon_github" href="https://github.com/intel-retail/order-accuracy/tree/main/dine-in">
     GitHub
  </a>
  <a class="icon_document" href="https://github.com/intel-retail/order-accuracy/blob/main/dine-in/README.md">
     Readme
  </a>
</div>
hide_directive-->

Staff-triggered plate validation workflow designed for full-service restaurant expo operations. This solution demonstrates zero-training deployment with vision-language models (VLM), semantic order reconciliation, and latency instrumentation aligned to operational service windows.

The Dine-In Order Accuracy solution enables restaurant staff to validate food plates against customer orders before serving. Unlike take-away scenarios with conveyor-based automation, dine-in operations require manual triggering by expo staff when plates are ready for service.

## Use Case

In a full-service restaurant:

1. Kitchen prepares a dish for Table 12
2. Expo staff places the plate in the validation station
3. Staff triggers validation via Gradio UI or API
4. System analyzes plate contents using VLM
5. System compares detected items against the order manifest
6. Staff receives immediate feedback on order accuracy

## Key Features

| Feature                      | Description                                                                      |
| ---------------------------- | -------------------------------------------------------------------------------- |
| **Zero-Training Deployment** | Uses pre-trained Qwen2.5-VL-7B model - no fine-tuning required                   |
| **Semantic Matching**        | Fuzzy item matching handles naming variations (e.g., "Big Mac" ↔ "Maharaja Mac") |
| **Real-Time Validation**     | Sub-15-second end-to-end latency for operational efficiency                      |
| **Circuit Breaker Pattern**  | Fault-tolerant with automatic service recovery                                   |
| **Connection Pooling**       | Optimized HTTP/2 connections for high throughput                                 |
| **Bounded Caching**          | LRU cache prevents memory exhaustion under load                                  |
| **Comprehensive Metrics**    | CPU, GPU, memory utilization with token-level inference stats                    |

## Next Steps

- [System Requirements](./get-started/system-requirements.md) - Check the detailed requirements
- [Installation Guide](./get-started.md) - Step-by-step installation instructions
- [How It Works](./how-it-works.md) - Learn about the architecture
- [How To Use](./how-to-use.md) - Learn how to operate the system
- [Benchmarking Guide](./di-benchmarking.md) - Run benchmarks
- [API Reference](./api-reference.md) - Learn the API
- [Troubleshooting](./troubleshooting.md) - Resolve common issues
- [Release Notes](./release-notes.md) - Read about updates and improvements

<!--hide_directive
:::{toctree}
:hidden:

./get-started.md
./how-it-works.md
How To Use <./how-to-use.md>
Benchmarking <./di-benchmarking.md>
API Reference <./api-reference.md>
Troubleshooting <./troubleshooting.md>
Release Notes <./release-notes.md>

:::
hide_directive-->
