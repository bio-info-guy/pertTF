# Ported over from scGPT repo:

 https://github.com/bowang-lab/scGPT

 ## this is an attempt to remove direct dependecies on the scGPT project
 - scGPT calls on torchtext for Vocab implementation, which is no longer maintained
 - flash_attention requires < 1.0.5 with many dependency issues