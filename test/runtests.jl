using LearnBenchmark
using Test

@testset "LearnBenchmark.jl" begin

    @testset "countpoints" begin
        @testset "1D" begin
            dataset = zeros(1,10)
            dataset[1,:] .= 1:10

            @test LearnBenchmark.countpoints([1],1.1,dataset) == 2
            @test LearnBenchmark.countpoints([2],1.1,dataset) == 3
            @test LearnBenchmark.countpoints([3],4.1,dataset) == 5
        end

        @testset "2D" begin
            dataset = zeros(2,100)
            dataset[1,:] .= repeat(1:10,10)
            dataset[2,:] .= repeat(1:10,inner = 10)

            @test LearnBenchmark.countpoints([2;2],1.1,dataset) == 5
            
        end
    end

    @testset "density_ratio_estimator!" begin
        data1 = zeros(1,10)
        data1[1:10] .= 1:10
        data2 = zeros(1,10)
        data2[1:10] = 1:10
        dataset = [data1, data2]

        ratios = zeros(2)
        center = [3]
        LearnBenchmark.density_ratio_estimator!(ratios,center, 1.5, dataset)
        @test ratios[1] ≈ 1

        data2[1:10] = 1:.5:5.5
        dataset = [data1, data2]
        ratios = zeros(2)
        LearnBenchmark.density_ratio_estimator!(ratios,center, 1.6^2, dataset)
        @test ratios[1] ≈ 3/7
    end

    @testset "Chebyshev Polynomials" begin
        @testset "chebyshev_poly" begin
            # First polynomial
            @test LearnBenchmark.chebyshev_poly(1, 0) == 1

            # Second
            @test LearnBenchmark.chebyshev_poly(.5, 1) == .5
            @test LearnBenchmark.chebyshev_poly(1, 1) == 1

            # Third f(x) = 2x^2 - 1
            @test LearnBenchmark.chebyshev_poly(.5, 2) == -.5
            @test LearnBenchmark.chebyshev_poly(1, 2) == 1
        end

        @testset "chebyshev_roots" begin
            @test all(LearnBenchmark.chebyshev_roots(1,2) .≈ [1])
            @test all(LearnBenchmark.chebyshev_roots(2,2) .≈ [1+sqrt(2)/2,1-sqrt(2)/2])
        end

        @testset "chebyshev_weights" begin
            @test sum(LearnBenchmark.chebyshev_weights(5,2)) ≈ 1
            weight_test = LearnBenchmark.chebyshev_weights(6,3)
            root_test = LearnBenchmark.chebyshev_roots(6)
            @test sum(weight_test .* root_test) ≈ 0 atol=1e-14
        end
    end


end
