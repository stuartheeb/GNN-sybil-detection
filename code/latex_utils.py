import os
import random

template_directory = "latex_templates"


def generate_plot(plots: [(str, str)],
                  preset=None,
                  target_directory="../thesis/plots",
                  target_file_name="plot.tex",
                  width="12cm",
                  height="8cm",
                  xmin="0",
                  xmax="10000",
                  ymin="0.0",
                  ymax="1.0",
                  xtick="1000,2000,...,10000",
                  ytick="0.0,0.1,...,1.0",
                  legend_position="south east",
                  ylabel="AUC",
                  xlabel="Number of attack edges",
                  caption="caption",
                  figure_label=None
                  ):
    if preset is not None:
        if preset == "AUC":
            ymin = "0.0"
            ymax = "1.0"
            ytick = "0.0,0.1,...,1.0"
        elif preset == "accuracy":
            ymin = "0.0"
            ymax = "1.0"
            ytick = "0.0,0.1,...,1.0"
        else:
            raise Exception("Unknown plot preset")

    content = []
    with open(f"{template_directory}/plot.tex") as file:
        for line in file:
            content.append(line)

    plot_strings = ""
    for (plot_file_name, plot_legend) in plots:
        temp = "\\addplot table {_FILENAME_};\n\\addlegendentry{_LEGENDENTRY_}\n"
        temp = temp.replace("_FILENAME_", plot_file_name)
        temp = temp.replace("_LEGENDENTRY_", plot_legend)
        plot_strings += temp

    output = []
    for line in content:
        line = line.replace("_WIDTH_", width)
        line = line.replace("_HEIGHT_", height)

        line = line.replace("_XMIN_", xmin)
        line = line.replace("_XMAX_", xmax)
        line = line.replace("_YMIN_", ymin)
        line = line.replace("_YMAX_", ymax)

        line = line.replace("_XTICK_", xtick)
        line = line.replace("_YTICK_", ytick)

        line = line.replace("_LEGENDPOS_", legend_position)

        line = line.replace("_YLABEL_", ylabel)
        line = line.replace("_XLABEL_", xlabel)

        line = line.replace("_PLOTS_", plot_strings)

        line = line.replace("_CAPTION_", caption)

        line = line.replace("_FIGLABEL_", figure_label if figure_label is not None else str(random.randint(0, 1000)))

        output.append(line)

    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    with open(f"{target_directory}/{target_file_name}", 'w') as file:
        for line in output:
            file.write(line)


def write_latex_figure_file(directory: str,
                            figure_file_name: str,
                            width: float = 0.5,
                            caption: str = "",
                            label: str = "figure",
                            float_specifier: str = "htbp"):
    to_write = "\\begin{figure}_FLOAT_SPECIFIER_\n\t\centering\n\t\includegraphics[width=_WIDTH_\\textwidth]{_FIGURE_FILE_}\n\t\caption{_CAPTION_}\n\t\label{fig:_LABEL_}\n\end{figure}"

    to_write = to_write.replace("_FIGURE_FILE_", f"{directory}/{figure_file_name}")
    to_write = to_write.replace("_WIDTH_", str(width))
    to_write = to_write.replace("_CAPTION_", caption)
    to_write = to_write.replace("_LABEL_", label)
    to_write = to_write.replace("_FLOAT_SPECIFIER_", f"[{float_specifier}]")

    with open(f"{directory}/{figure_file_name.split('.')[0]}.tex", 'w') as file:
        file.write(to_write)
