module LearnBenchmark


export bayeserror

#########################################################
#########################################################
################# Ensemble Learner ######################
#########################################################
#########################################################

"""
    ensemblelearner(X, L, bound_ratio)

### Arguments
 - `X`              -   Data as an array of λ matrices of shape M × Nᵢ
 - `L`              -   Number of ϵ-balls
 - `bound_ratio`    -   Ratio of density bounds CL/CU

### Returns
An estimate of the Bayes error based on ϵ-balls
"""
function ensemblelearner(X, L, bound_ratio)
    N = [size(data)[2] for data in X]
    d = size(X[1])[1]
    λ = length(N)

    prior = N./sum(N)
    ϵ² = chebyshev_roots(L).^2 ./ (N[1] .^(1/d))
    weights = chebyshev_weights(L, d)
    reverse!(weights)

    ratios = zeros(λ,L)
    
    estimate = 0

    # For Each Class
    for l in 2:λ

        # For each datapoint in the class
        for i in 1:N[l]
            # Set center of ϵ-ball
            center = @view X[l][:,i]

            # For each radius
            for k in 1:L
                # Compute estimates of density ratios
                density_ratio_estimator!(   view(ratios,1:l,k),
                                            center, 
                                            ϵ²[k], 
                                            view(X[1:l],1:l))
            end

            # Weight Ratios
            weighted_ratios = ratios * weights

            # update estimate
            estimate += tₖ(     prior[1:l], 
                                weighted_ratios[1:l]) / N[l]
            #estimate += tₖ_bar( prior[1:l], 
            #                    weighted_ratios[1:l], 
            #                    bound_ratio) / N[l]
        end
    end

    return 1 - prior[1] - estimate
end


"""
    bayeserror(X, L, bound_ratio)

### Arguments
 - `X`              -   Data as an array of λ matrices of shape M × Nᵢ
 - `L`              -   Number of ϵ-balls
 - `bound_ratio`    -   Ratio of density bounds CL/CU

### Returns
An estimate of the Bayes error based on ϵ-balls
"""
bayeserror(X, L, bound_ratio) = ensemblelearner(X, L, bound_ratio)


#########################################################
#########################################################
##################### Weak Learner ######################
#########################################################
#########################################################

"""
    countpoints(center, ϵ², dataset)

Counts the number of points in the intersection of a dataset
and an ϵ-ball specified by its center and radius.

### Arguments
 - `center` - Center of ϵ-ball
 - `ϵ²`     - Squared radius of ϵ-ball
 - `dataset`- Dataset to count, N × M matrix with N different points

### Returns
The number of points from `dataset` within the specified ϵ-ball
"""
function countpoints(center, ϵ², dataset)
    M, N = size(dataset)
    count = 0

    for i in 1:N 
        if sum((x) -> x^2, dataset[:,i] - center) < ϵ²
            count += 1
        end
    end

    return count
end

"""
    density_ratio_estimator!(ratios,center, ϵ², dataset)

Construct the set of density ratio estimators for a given ϵ-ball.
This corresponds to the weak learner in this algorithm

### Arguments
 - `ratios` - Stores density ratio estimates
 - `center` - Center of the ϵ-ball 
 - `ϵ²`     - Defines radius of ϵ-ball
 - `dataset`- Dataset for learner
"""
function density_ratio_estimator!(ratios,center, ϵ², dataset)
    l = size(dataset)[1]

    # For each previous class, count points
    for j in 1:l
        ratios[j] = countpoints(center,ϵ²,dataset[j])
    end

    # Normalize to construct ratios
    ratios[1:l-1] ./= ratios[l]
end

#########################################################
#########################################################
################ Divergence Functions ###################
#########################################################
#########################################################

"""
    tₖ(prior,data)

Divergence function used in the paper

### Arguments
 - `prior`  - prior distribution based on number of points
 - `data`   - function inputs based on local density
"""
function tₖ(prior,data)
    return max(0, prior[end] - maximum(prior[1:end-1] .* data[1:end-1]))
end

"""
    tₖ_bar(prior, data, bound_ratio)

Divergence function used in the paper

### Arguments
 - `prior`          - prior distribution based on number of points
 - `data`           - function inputs based on local density
 - `bound_ratio`    - Ratio of density bounds CL/CU
"""
function tₖ_bar(prior, data, bound_ratio = 1e-4)
    bound_ratio_array = [bound_ratio for i in 1:length(prior)]

    return max(     tₖ(prior,data),
                    tₖ(prior,bound_ratio_array))
end


#########################################################
#########################################################
################# Weight Computation ####################
#########################################################
#########################################################

"""
    chebyshev_weights(L, d)

Computes the optimal weights for the ensemble learner
"""
function chebyshev_weights(L, d, α=.5)
    @assert L ≥ d

    roots = chebyshev_roots(L, α)

    weights = zeros(L)
    weights .-= 1/L
    for i in 1:L
        for k in 0:d
            tmp  = chebyshev_poly_scaled(0,k, α)
            tmp *= chebyshev_poly_scaled(roots[i],k, α)
            tmp *= 2/L

            weights[i] += tmp
        end
    end

    return weights
end


"""
    chebyshev_roots(L)

Computes the roots of an L+1'th order Chebyshev polynomial
scaled horizontally by a factor of 2
"""
function chebyshev_roots(L, α=.5)
    roots = zeros(L)
    for i in 1:L
        roots[i] = cos((i - 0.5) * π/L)*α/2 + α/2
    end
    return roots
end

"""
    chebyshev_poly_scaled(x, k)

Evaluates the k'th Chebyshev polynomial at x after horizontal scaling by 2 and shifting by 1
"""
function chebyshev_poly_scaled(x, k, α=.5)
    return chebyshev_poly(x*2/α - 1, k)
end

"""
chebyshev_poly(x, k)

Evaluates the k'th Chebyshev polynomial at x
"""
function chebyshev_poly(x, k)
    if  k == 0
        return 1 
    end
    d = 1
    out = x
    out_old = 1
    for i = d+1:k
        tmp = out
        out = 2*x*out - out_old
        out_old = tmp
    end
    return out
end

end