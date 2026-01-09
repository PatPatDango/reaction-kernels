Please use option functions rsmi_to_graph and rsmi_to_its  with options drop_non_aam=False and use_index_as_atom_map=False


i.e. educt_graph, product_graph = rsmi_to_graph(smiles, drop_non_aam=False, use_index_as_atom_map=False)
and
its_graph = rsmi_to_its(smiles, drop_non_aam=False, use_index_as_atom_map=False