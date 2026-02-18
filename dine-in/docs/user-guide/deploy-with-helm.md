# Deploy with Helm

Instructions for deploying Dine-In Order Accuracy using Helm on Kubernetes.

## Prerequisites

- Kubernetes cluster (1.24+)
- Helm 3.10+
- kubectl configured with cluster access
- Intel GPU device plugin (for GPU acceleration)

## Helm Chart Structure

```
helm/
├── Chart.yaml
├── values.yaml
└── templates/
    ├── deployment.yaml
    ├── service.yaml
    ├── configmap.yaml
    └── ingress.yaml
```

## Installation

### Add Helm Repository

```bash
helm repo add order-accuracy https://charts.example.com/order-accuracy
helm repo update
```

### Install Chart

```bash
helm install dine-in order-accuracy/dine-in \
  --namespace order-accuracy \
  --create-namespace
```

### Install with Custom Values

```bash
helm install dine-in order-accuracy/dine-in \
  --namespace order-accuracy \
  --create-namespace \
  --set ovms.replicas=2 \
  --set app.resources.limits.memory=4Gi
```

## Configuration

### values.yaml

```yaml
# Application settings
app:
  replicaCount: 1
  image:
    repository: dine-in-dine-in
    tag: latest
    pullPolicy: IfNotPresent
  resources:
    limits:
      cpu: "2"
      memory: 4Gi
    requests:
      cpu: "1"
      memory: 2Gi
  service:
    type: ClusterIP
    ports:
      ui: 7861
      api: 8083

# OVMS settings
ovms:
  replicaCount: 1
  image:
    repository: openvino/model_server
    tag: latest
  resources:
    limits:
      gpu.intel.com/i915: 1
      memory: 16Gi
    requests:
      memory: 8Gi
  modelPath: /models/vlm

# Semantic service
semantic:
  enabled: true
  replicaCount: 1
  image:
    repository: semantic-comparison-service
    tag: latest

# Ingress
ingress:
  enabled: false
  className: nginx
  hosts:
    - host: dine-in.local
      paths:
        - path: /
          pathType: Prefix
```

## Verify Deployment

```bash
# Check pods
kubectl get pods -n order-accuracy

# Check services
kubectl get svc -n order-accuracy

# View logs
kubectl logs -f deployment/dine-in -n order-accuracy
```

## Port Forwarding (Development)

```bash
# Gradio UI
kubectl port-forward svc/dine-in 7861:7861 -n order-accuracy

# REST API
kubectl port-forward svc/dine-in 8083:8083 -n order-accuracy
```

## Upgrade

```bash
helm upgrade dine-in order-accuracy/dine-in \
  --namespace order-accuracy \
  --set app.image.tag=v2.0.0
```

## Uninstall

```bash
helm uninstall dine-in -n order-accuracy
```

## GPU Support

### Intel GPU Device Plugin

Ensure Intel GPU device plugin is installed:

```bash
kubectl apply -f https://raw.githubusercontent.com/intel/intel-device-plugins-for-kubernetes/main/deployments/gpu_plugin/gpu_plugin.yaml
```

### Verify GPU Access

```bash
kubectl get nodes -o json | jq '.items[].status.allocatable["gpu.intel.com/i915"]'
```

## Troubleshooting

### Pod Not Starting

```bash
kubectl describe pod <pod-name> -n order-accuracy
kubectl logs <pod-name> -n order-accuracy
```

### GPU Not Detected

```bash
# Check device plugin
kubectl get pods -n kube-system | grep intel-gpu

# Check node labels
kubectl get nodes --show-labels | grep gpu
```

### Service Unreachable

```bash
# Check endpoints
kubectl get endpoints -n order-accuracy

# Test from inside cluster
kubectl run test --rm -it --image=curlimages/curl -- curl http://dine-in:8083/health
```
