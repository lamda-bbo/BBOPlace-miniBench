import os
from utils.constant import INF
from utils.read_benchmark.rules import RegExRule, RuleGroup, AlwaysMatchedRule, AlwaysDismatchedRule, SkipRule, EntryFormatWithRuleGroups

def read_benchmark(design_name, benchmark_path, n_macro):
    node_file = open(os.path.join(benchmark_path, f"{design_name}.nodes"), 'r')
    all_node_info, node_info_raw_id_name, node_total_area = read_node_file(node_file)

    # select macro
    n_macro = min(len(all_node_info), n_macro)
    all_node_lst = list(all_node_info.keys())
    all_node_lst = sorted(all_node_lst, key=lambda x:all_node_info[x]["size_x"] * all_node_info[x]["size_y"], reverse=True)
    node_info = {}
    for macro_id in range(n_macro):
        macro = all_node_lst[macro_id]
        node_info[macro] = all_node_info[macro]

    node_cnt = len(node_info)
    node_file.close()

    net_file = open(os.path.join(benchmark_path, f"{design_name}.nets"), 'r')
    net_info = read_net_file(net_file, node_info)
    net_cnt = len(net_info)
    net_file.close()

    pl_file = open(os.path.join(benchmark_path, f"{design_name}.pl"), 'r')
    canvas_ux, canvas_uy, canvas_lx, canvas_ly, cell_name = read_pl_file(pl_file, node_info)
    pl_file.close()

    port_info = {}
    port_to_net_dict = {}

    scl_file = os.path.join(benchmark_path, f"{design_name}.scl")
    corerow_info_lst, num_rows = read_scl(scl_file)
    canvas_lx = sorted(corerow_info_lst, key=lambda x:x["subrow_origin"])[0]["subrow_origin"]
    canvas_ly = sorted(corerow_info_lst, key=lambda x:x["coor"])[0]["coor"]
    canvas_ux = sorted(corerow_info_lst, key=lambda x:x["num_sites"], reverse=True)[0]["num_sites"] + canvas_lx
    canvas_uy = sorted(corerow_info_lst, key=lambda x:x["height"], reverse=True)[0]["height"] * num_rows + canvas_ly
    placedb_info = {
        'node_info' : node_info,
        'node_info_raw_id_name' : node_info_raw_id_name,
        'node_cnt' : node_cnt,
        'port_info' : port_info,
        'net_info' : net_info,
        'net_cnt' : net_cnt,
        'canvas_lx' : canvas_lx,
        'canvas_ly' : canvas_ly,
        'canvas_ux' : canvas_ux,
        'canvas_uy' : canvas_uy, 
        'cell_name' : cell_name,
        'port_to_net_dict' : port_to_net_dict,
        'node_total_area' : node_total_area,
        'database' : None,
    }
    return placedb_info


def read_node_file(fopen):
    node_info = {} 
    node_info_raw_id_name ={}
    node_cnt = 0
    cell_total_area = 0
    for line in fopen.readlines():
        if not line.startswith("\t"):
            continue
        line = line.strip().split()

        # compute cell_total_area
        cell_area = int(line[1]) * int(line[2])
        cell_total_area += cell_area
        if line[-1] != "terminal":
            continue
        node_name = line[0]
        size_x = int(line[1])
        size_y = int(line[2])
        node_info[node_name] = {"id": node_cnt, "size_x": size_x , "size_y": size_y, "area": cell_area}
        node_info_raw_id_name[node_cnt] = node_name
        node_cnt += 1
    # print("len node_info", len(node_info))
    return node_info, node_info_raw_id_name, cell_total_area

def read_net_file(fopen, node_info):
    net_info = {}
    net_name = None
    net_cnt = 0
    for line in fopen.readlines():
        if not line.startswith("\t") and not line.startswith("NetDegree"):
            continue
        line = line.strip().split()
        if line[0] == "NetDegree":
            net_name = line[-1]
        else:
            node_name = line[0]
            if node_name in node_info:
                if not net_name in net_info:
                    net_info[net_name] = {}
                    net_info[net_name]["nodes"] = {}
                    net_info[net_name]["ports"] = {}
                if not node_name in net_info[net_name]["nodes"]:
                    x_offset = float(line[-2])
                    y_offset = float(line[-1])
                    net_info[net_name]["nodes"][node_name] = {}
                    net_info[net_name]["nodes"][node_name] = {"x_offset": x_offset, "y_offset": y_offset}
    
    for net_name in list(net_info.keys()):
        if len(net_info[net_name]["nodes"]) <= 1:
            net_info.pop(net_name)
    for net_name in net_info:
        net_info[net_name]['id'] = net_cnt
        net_cnt += 1
    # print("adjust net size = {}".format(len(net_info)))
    return net_info

def read_pl_file(fopen, node_info):
    standard_cell_name = []
    max_height = 0
    max_width = 0
    min_height = INF
    min_width = INF
    for line in fopen.readlines():
        if not line.startswith('o'):
            continue
        line = line.strip().split()
        node_name = line[0]
        if not node_name in node_info:
            standard_cell_name.append(node_name)
            continue
        place_x = int(line[1])
        place_y = int(line[2])
        max_width = max(max_width, node_info[node_name]["size_x"] + place_x)
        max_height = max(max_height, node_info[node_name]["size_y"] + place_y)
        min_width = min(min_width, place_x)
        min_height = min(min_height, place_y)
        node_info[node_name]["raw_x"] = place_x
        node_info[node_name]["raw_y"] = place_y
    return max_width, max_height, min_width, min_height, standard_cell_name

def read_scl(scl_file):
    scl_origin = [[]]
    numrow_rule = RegExRule(
        r"NumRows\s*:\s*(\d+)",
        {
            "num_rows" : 1,
        }
    )
    row_start_rule = RegExRule(
        r"(CoreRow)\s*Horizontal",
        {
            "start" : 1,
        }
    )
    row_end_rule = RegExRule(
        r"(End)",
        {
            "end" : 1,
        }
    )
    coor_rule = RegExRule(
        r"\t*Coordinate\s*:\s*(\d+)",
        {
            "coor" : 1,
        }
    )
    height_rule = RegExRule(
        r"\t*Height\s*:\s*(\d+)",
        {
            "height" : 1,
        }
    )
    sitewidth_rule = RegExRule(
        r"\t*Sitewidth\s*:\s*(\d+)",
        {
            "site_width" : 1,
        }
    )
    sitespacing_rule = RegExRule(
        r"\t*Sitespacing\s*:\s*(\d+)",
        {
            "site_spacing" : 1,
        }
    )
    siteorient_rule = RegExRule(
        r"\t*Siteorient\s*:\s*(\d+)",
        {
            "site_orient" : 1,
        }
    )
    sitesymmetry_rule = RegExRule(
        r"\t*Sitesymmetry\s*:\s*(\d+)",
        {
            "site_symmetry" : 1,
        }
    )
    subroworigin_rule = RegExRule(
        r"\t*SubrowOrigin\s*:\s*(\d+)\s*NumSites\s*:\s*(\d+)",
        {
            "subrow_origin" : 1,
            "num_sites" : 2,
        }
    ) 
    core_row_group = RuleGroup(
        row_start_rule,
        row_end_rule,
        [
            row_start_rule,
            row_end_rule,
            coor_rule,
            height_rule,
            sitewidth_rule,
            sitespacing_rule,
            siteorient_rule,
            sitesymmetry_rule,
            subroworigin_rule
        ]
    )
    other_rule_group = RuleGroup(
        
        AlwaysMatchedRule(),
        AlwaysMatchedRule(),
        [
            numrow_rule,
            SkipRule()
        ]
    )
    scl_ent_format = EntryFormatWithRuleGroups(
        [
            core_row_group,
            other_rule_group
        ]
    )

    assert scl_file is not None and os.path.exists(scl_file)

    corerow_info_lst = []
    with open(scl_file, "r") as f:
        for line in f:
            output = scl_ent_format(line)
            # print(line, output)
            if output.get("entry", None) is not None:
                entry = output.get("entry")
                scl_origin[-1].append("%s" % (entry))
            else:
                scl_origin.append([])
            if output == {}:
                continue
            
            if "num_rows" in output.keys():
                num_rows = int(output["num_rows"])
            elif "start" in output.keys():
                row_info = {}
            elif "end" in output.keys():
                assert len(row_info) == 8
                corerow_info_lst.append(row_info)
                del row_info
            elif "coor" in output.keys():
                row_info["coor"] = int(output["coor"])
            elif "height" in output.keys():
                row_info["height"] = int(output["height"])
            elif "site_width" in output.keys():
                row_info["site_width"] = int(output["site_width"])
            elif "site_spacing" in output.keys():
                row_info["site_spacing"] = int(output["site_spacing"])
            elif "site_orient" in output.keys():
                row_info["site_orient"] = int(output["site_orient"])
            elif "site_symmetry" in output.keys():
                row_info["site_symmetry"] = int(output["site_symmetry"])
            elif "subrow_origin" in output.keys():
                row_info["subrow_origin"] = int(output["subrow_origin"])
                row_info["num_sites"] = int(output["num_sites"])
    
    assert len(corerow_info_lst) == num_rows

    for i in range(len(scl_origin)):
        scl_origin[i] = "".join(scl_origin[i])

    return corerow_info_lst, num_rows