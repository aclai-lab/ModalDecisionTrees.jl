# Changelog

This is a changelog containing both logical, structural, educational, and stylistical considerations that emerged while refactoring version 0.5.2 of this package.

*Logical* considerations are related to code functioning (e.g., keeping note of what must be tested, what functionalities are redundant, etc.)

*Educational* considerations are related to docstrings and usage experience.

*Structural* considerations are about the structure of the project.

*Style* considerations are related to little stylistic changes, such as replacing `my_foo` with `myfoo` or avoiding unnecessary tabs. (Actually, as ACLAI, we should think about choosing one definitive Julia style).

# Structural considerations
- @mauro tried to tidy up a bit the using/import and export keywords in ModalDecisionTrees.jl

# Style considerations

- Little utilities, which do not require extensive documentation and are not exported, might start with "_"; for example, `partition!` is used only one in the whole package, and could hence be substituted with `_partition!`.