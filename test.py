from data4robotics.models.resnet import ResNet

m = ResNet(size=34, weights='IMAGENET1K_V1', norm_cfg=dict(name='batch_norm'))

