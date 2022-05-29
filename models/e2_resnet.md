## Doubts about equivariant ResNet
I'm not sure why in the WideBasic block we have `in_fiber`, `inner_fiber` and `out_fiber` while in BasicBlock there is just `in_planes` and `out_planes`. 
I think maybe it is related to `self.convShorcut` in BasicBlock.


I want to build an equivariant bottleneck class. For that I need to build a FieldType that contains `width` channels.
Is that the correct way to do it?

```python
# I think len(out_fiber) is the number of channels in E2
planes = len(out_fiber)
# for ResNext50_32x4d base_width (width_per_group) is 4 and there are 32 groups
width = int(planes * (base_width / 64.)) * groups

first_rep_type = type(in_fiber.representations[0])
for rep in in_fiber.representations:
    assert first_rep_type == type(rep)

# Here the representation of in_fiber is hardcoded, is there some property that allows me
# to retrieve from in_fiber?
in_rep = 'regular'
width_fiber = nn.FieldType(in_fiber.gspace, width * [in_fiber.gspace.representations[in_rep]])
```

Not sure why we use conv3x3 for rotations 0, 2 and 4 and conv5x5 for other rotations


At the end initializing the network, is this init correct?
```python
elif isinstance(module, torch.nn.BatchNorm2d):
    module.weight.data.fill_(1)
    module.bias.data.zero_()
elif isinstance(module, torch.nn.Linear):
    module.bias.data.zero_()
```
BatchNorm2d isn't even used, should we replace it with InnerBatchNorm or remove that part entirely 
(InnerBatchNorm doesn't have instance variables `weight` or `bias` from what I've checked). 
Also why is the standard linear initialized to 0?