using SoleData.DimensionalDatasets

const ALLOW_GLOBAL_SPLITS = true

const mlj_default_max_depth = nothing
const mlj_default_max_modal_depth = nothing

const mlj_mdt_default_min_samples_leaf = 4
const mlj_mdt_default_min_purity_increase = 0.002
const mlj_mdt_default_max_purity_at_leaf = Inf
const mlj_mdt_default_n_subfeatures = identity

const mlj_mrf_default_min_samples_leaf = 1
const mlj_mrf_default_min_purity_increase = -Inf
const mlj_mrf_default_max_purity_at_leaf = Inf
const mlj_mrf_default_ntrees = 50
sqrt_f(x) = ceil(Int, sqrt(x))
const mlj_mrf_default_n_subfeatures = sqrt_f
const mlj_mrf_default_sampling_fraction = 0.7

mlj_default_initconditions = nothing

mlj_default_initconditions_str = "" *
    ":start_with_global" # (i.e., starting with a global decision, such as ⟨G⟩ min(V1) > 2) " *
    # "for 1-dimensional data and :start_at_center for 2-dimensional data."

AVAILABLE_INITCONDITIONS = OrderedDict{Symbol,InitialCondition}([
    :start_with_global => MDT.start_without_world,
    :start_at_center   => MDT.start_at_center,
])


function readinitconditions(model, dataset)
    if SoleData.ismultilogiseed(dataset)
        map(mod->readinitconditions(model, mod), eachmodality(dataset))
    else
        if model.initconditions == mlj_default_initconditions
            # d = dimensionality(SoleData.base(dataset)) # ? TODO maybe remove base for AbstractModalLogiset's?
            d = dimensionality(frame(dataset, 1))
            if d == 0
                AVAILABLE_INITCONDITIONS[:start_with_global]
            elseif d == 1
                AVAILABLE_INITCONDITIONS[:start_with_global]
            elseif d == 2
                AVAILABLE_INITCONDITIONS[:start_with_global]
            else
                error("Unexpected dimensionality: $(d)")
            end
        else
            AVAILABLE_INITCONDITIONS[model.initconditions]
        end
    end
end
