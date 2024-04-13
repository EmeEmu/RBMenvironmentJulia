using Pkg
Pkg.activate("./")

using RestrictedBoltzmannMachines: BinaryRBM, sample_v_from_v
using CudaRBMs: cpu, gpu
using Random

rbm = BinaryRBM(randn(5), randn(3), randn(5, 3))
rbm_cuda = gpu(rbm)

v = rand(Float64, 5, 10) .> 0.5
v_cuda = gpu(v)

v_samp_cuda = sample_v_from_v(rbm_cuda, v_cuda)
v_samp = gpu(v_samp_cuda)

println(v_samp)
