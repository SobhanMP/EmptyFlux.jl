using ProgressLogging: @progress, @withprogress, @logprogress


"""
    update!(opt, p, g)

Perform an update step of the parameters `ps` (or the single parameter `p`)
according to optimizer `opt`  and the gradients `gs` (the gradient `g`).

As a result, the parameters are mutated and the optimizer's internal state may change.
The gradient could be mutated as well.
"""
function update!(opt::AbstractOptimiser, x, x̄)
  x̄r = ArrayInterface.restructure(x, x̄) # address some cases where Zygote's
                                          # output are not mutable, see #1510 
  x .-= apply!(opt, x, x̄r)
end


# Callback niceties
call(f, xs...) = f(xs...)
runall(f) = f
runall(fs::AbstractVector) = () -> foreach(call, fs)

struct SkipException <: Exception end

"""
    skip()

Call `Flux.skip()` in a callback to indicate when a callback condition is met.
This will trigger the train loop to skip the current data point and not update with the calculated gradient.

# Examples
```julia
cb = function ()
  loss() > 1e7 && Flux.skip()
end
```
"""
function skip()
  throw(SkipException())
end


struct StopException <: Exception end

"""
    stop()

Call `Flux.stop()` in a callback to indicate when a callback condition is met.
This will trigger the train loop to stop and exit.

# Examples
```julia
cb = function ()
  accuracy() > 0.9 && Flux.stop()
end
```
"""
function stop()
  throw(StopException())
end

batchmemaybe(x) = tuple(x)
batchmemaybe(x::Tuple) = x


"""
    @epochs N body

Run `body` `N` times. Mainly useful for quickly doing multiple epochs of
training in a REPL.

# Examples
```jldoctest
julia> Flux.@epochs 2 println("hello")
[ Info: Epoch 1
hello
[ Info: Epoch 2
hello
```
"""
macro epochs(n, ex)
  :(@progress for i = 1:$(esc(n))
      @info "Epoch $i"
      $(esc(ex))
    end)
end
