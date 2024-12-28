# Faster R-CNN ResNet-50 с FPN (Feature Pyramid Network)

ResNet-50 is a convolutional neural network (CNN) known for its deep architecture and efficiency. It excels at extracting features from images, which is key for object detection tasks.

When you call `fasterrcnn_resnet50_fpn` in `torchvision`, the model automatically loads the pre-trained weights (if `pretrained=True` is specified), allowing you to use it "out of the box" for object detection tasks in images

docker compose

```yml
---
services:
  coco2:
    image: "ghcr.io/rootshell-coder/person-detector:latest"
    runtime: "nvidia-runc"
    ports:
      - "5000:5000"
    networks:
      - "pd-net"
    restart: "always"
    deploy:
      resources:
        limits:
          cpus: "6.0"
          memory: "2G"
        reservations:
          cpus: "1.0"
          memory: "2G"
networks:
  pd-net:
    name: "pd-net"
```

Remove runtime: `runtime: "nvidia-runc"` if you do not know what is it or not CUDA support in the system.

Check my wedding photo directory and select all where a person is present.

```bash
photo_dir=/home/RootShell-coder/wedding

for file in $(find ${photo_dir} -type f -name "*.jpg"); do
  echo "Sending $file"
  curl -s -X POST -F "file=@$file" -F "full_path=$(pwd)/$file" http://localhost:5000/detect
done
```

In response, we receive a JSON that can already be partially used to sort photos
