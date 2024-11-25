
using SoleData
using SoleData: AbstractModalLogiset, SupportedLogiset

using MultiData
using MultiData: dataframe2dimensional

function wrapdataset(
    X,
    model,
    force_var_grouping::Union{Nothing,AbstractVector{<:AbstractVector}} = nothing;
    passive_mode = false,
)
    SoleData.autologiset(
        X;
        force_var_grouping = force_var_grouping,
        downsize = model.downsize,
        conditions = model.conditions,
        featvaltype = model.featvaltype,
        relations = model.relations,
        fixcallablenans = model.fixcallablenans,
        passive_mode = passive_mode,
    )
end
