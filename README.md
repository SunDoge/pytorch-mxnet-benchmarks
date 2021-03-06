# Pytorch MXNet Benchmarks

## Version

| Framework | Version                        |
| --------- | ------------------------------ |
| Pytorch   | py3.6_cuda9.0.176_cudnn7.4.1_1 |
| MXNet     | mxnet-cuda90 1.3.1             |

## Result

### Memory Usage

Batch Size: 128

| Model                    | Usage              |
| ------------------------ | ------------------ |
| resnet20-pytorch         | 801MiB / 12196MiB  |
| resnet20-mxnet           | 1203MiB / 12196MiB |
| resnet20-mxnet-hybridize | 797MiB / 12196MiB  |

### Time

Epochs: 200

| Model                    | Time               |
| ------------------------ | ------------------ |
| resnet20-pytorch         | 2287.0581789016724 |
| resnet20-mxnet           | 2120.8806025981903 |
| resnet20-mxnet-hybridize | 2866.426700115204  |

### Accuracy

| Model                    | Accuracy          |
| ------------------------ | ----------------- |
| resnet20-pytorch         | 91.74             |
| resnet20-mxnet           | [TODO]            |
| resnet20-mxnet-hybridize | 91.71000000000001 |
