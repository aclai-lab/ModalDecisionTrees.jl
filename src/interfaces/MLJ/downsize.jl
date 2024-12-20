# using SoleData: make_downsizing_function
import SoleData: make_downsizing_function


function make_downsizing_function(::TreeModel)
    make_downsizing_function(Val(1))
end
function make_downsizing_function(::ForestModel)
    make_downsizing_function(Val(2))
end

function make_downsizing_function(::StumpsModel)
    make_downsizing_function(Val(1))
end
