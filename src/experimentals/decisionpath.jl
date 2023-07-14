using ..ModalDecisionTrees
using SoleModels
using SoleModels.DimensionalDatasets
using ..ModalDecisionTrees: AbstractFeature, TestOperator

using ..ModalDecisionTrees: ModalityId

using ..ModalDecisionTrees: DTLeaf, DTNode, DTInternal

import SoleModels: feature, test_operator, threshold

export DecisionPath, DecisionPathNode,
            get_path_in_tree, get_internalnode_dirname,
            mk_tree_path, get_tree_path_as_dirpath

struct DecisionPathNode
    taken         :: Bool
    feature       :: AbstractFeature
    test_operator :: TestOperator
    threshold     :: T where T
    worlds        :: AbstractWorldSet
end

taken(n::DecisionPathNode) = n.taken
feature(n::DecisionPathNode) = n.feature
test_operator(n::DecisionPathNode) = n.test_operator
threshold(n::DecisionPathNode) = n.threshold
worlds(n::DecisionPathNode) = n.worlds


const DecisionPath = Vector{DecisionPathNode}

_get_path_in_tree(leaf::DTLeaf, X::Any, i_instance::Integer, worlds::AbstractVector{<:AbstractWorldSet}, i_modality::ModalityId, paths::Vector{DecisionPath})::AbstractWorldSet = return worlds[i_modality]
function _get_path_in_tree(tree::DTInternal, X::MultiLogiset, i_instance::Integer, worlds::AbstractVector{<:AbstractWorldSet}, i_modality::Integer, paths::Vector{DecisionPath})::AbstractWorldSet
    satisfied = true
    (satisfied,new_worlds,worlds_map) =
        modalstep(
                        modality(X, i_modality(tree)),
                        i_instance,
                        worlds[i_modality(tree)],
                        decision(tree),
                        Val(true)
                    )

    worlds[i_modality(tree)] = new_worlds
    survivors = _get_path_in_tree((satisfied ? left(tree) : right(tree)), X, i_instance, worlds, i_modality(tree), paths)

    # if survivors of next step are in the list of worlds viewed by one
    # of the just accumulated "new_worlds" then that world is a survivor
    # for this step
    new_survivors::AbstractWorldSet = Vector{AbstractWorld}()
    for curr_w in keys(worlds_map)
        if length(intersect(worlds_map[curr_w], survivors)) > 0
            push!(new_survivors, curr_w)
        end
    end

    pushfirst!(paths[i_instance], DecisionPathNode(satisfied, feature(decision(tree)), test_operator(decision(tree)), thresholda(decision(tree)), deepcopy(new_survivors)))

    return new_survivors
end
function get_path_in_tree(tree::DTree{S}, X)::Vector{DecisionPath} where {S}
    _ninstances = ninstances(X)
    paths::Vector{DecisionPath} = [ DecisionPath() for i in 1:_ninstances ]
    for i_instance in 1:_ninstances
        worlds = ModalDecisionTrees.mm_instance_initialworldset(X, tree, i_instance)
        _get_path_in_tree(root(tree), X, i_instance, worlds, 1, paths)
    end
    paths
end

function get_internalnode_dirname(node::DTInternal)::String
    replace(displaydecision(node), " " => "_")
end

mk_tree_path(leaf::DTLeaf; path::String) = touch(path * "/" * string(prediction(leaf)) * ".txt")
function mk_tree_path(node::DTInternal; path::String)
    dir_name = get_internalnode_dirname(node)
    mkpath(path * "/Y_" * dir_name)
    mkpath(path * "/N_" * dir_name)
    mk_tree_path(left(node); path = path * "/Y_" * dir_name)
    mk_tree_path(right(node); path = path * "/N_" * dir_name)
end
function mk_tree_path(tree_hash::String, tree::DTree; path::String)
    mkpath(path * "/" * tree_hash)
    mk_tree_path(root(tree); path = path * "/" * tree_hash)
end

function get_tree_path_as_dirpath(tree_hash::String, tree::DTree, decpath::DecisionPath; path::String)::String
    current = root(tree)
    result = path * "/" * tree_hash
    for node in decpath
        if current isa DTLeaf break end
        result *= "/" * (node.taken ? "Y" : "N") * "_" * get_internalnode_dirname(current)
        current = node.taken ? left(current) : right(current)
    end
    result
end
