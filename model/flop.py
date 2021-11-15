import numpy as np

channel = [[3,64],[64,64],[64,128],[128,128], [128,256],[256,256],[256,256]]
temporal = [300,300,150,150,150,75,75]

# ST-GCN64,  64,128,  128,  256,  256  and  256.
Sflops = 0
Tflops = 0
for layer in range(len(channel)):
	Sflops += 3*(channel[layer][0]*channel[layer][1]*25 + channel[layer][1]*25*25)*temporal[layer]
	Tflops += 9*(temporal[layer]*25)*(channel[layer][1]*channel[layer][1])
fc = 256*60

print('ST-GCN')
print('spatial', Sflops*2/1e9)
print('temporal', Tflops*2/1e9)
print('1 stream', (Sflops+Tflops+fc)*2/1e9)
print('2 stream', (Sflops+Tflops+fc)/1e9*2*2)
print('4 stream', (Sflops+Tflops+fc)/1e9*2*4)