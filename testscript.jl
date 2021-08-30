using LearnBenchmark

d = 5
N = 5000
bound_ratio = .01

offset = zeros(d,N)
offset[1,:] .+= 1

dataset = [randn(d,N), randn(d,N) + offset]

L = d+1

@info "Ensemble Learner Results"
@info bayeserror(dataset,L,bound_ratio)
#println()
@info "Base Learner Results"
@info baselearner(dataset,3, bound_ratio)
