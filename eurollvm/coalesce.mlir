#map0 = affine_map<(d0, d1, d2, d3) -> (d0 * 393216 + d1 * 131072 + d2 * 256 + d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0 * 393216 + d1 * 131072 + d2 * 256 + d3 + 65536)> 

func.func @coalesce(%val : bf16, %bigmem : memref<1x3x512x256xbf16, 1>) {

  %smolmem0 = memref.subview %bigmem[0, 0, 0, 0] [1, 3, 256, 256] [1, 1, 1, 1] : memref<1x3x512x256xbf16, 1> to memref<1x3x256x256xbf16, #map0, 1>
  %smolmem1 = memref.subview %bigmem[0, 0, 256, 0] [1, 3, 256, 256] [1, 1, 1, 1] : memref<1x3x512x256xbf16, 1> to memref<1x3x256x256xbf16, #map1, 1>

  linalg.fill ins(%val : bf16) outs(%smolmem0 : memref<1x3x256x256xbf16, #map0, 1>)
  linalg.fill ins(%val : bf16) outs(%smolmem1 : memref<1x3x256x256xbf16, #map1, 1>)

  return
}
