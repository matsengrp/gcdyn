library(tidyverse)
library(coda)
library(glue)
library(patchwork)
library(kableExtra)

samples <- read_csv("samples.csv") |>
    rename(Iteration = `...1`)

if ("mutation_rate" %in% colnames(samples)) {
    samples <- pivot_longer(samples, c("xscale", "xshift", "yscale", "yshift", "death_rate", "mutation_rate"),
        names_to = "Parameter",
        values_to = "Sample"
    )
} else {
    samples <- pivot_longer(samples, c("xscale", "xshift", "yscale", "yshift", "death_rate"),
        names_to = "Parameter",
        values_to = "Sample"
    )
}

truth <- tibble(
    Parameter = c("xscale", "xshift", "yscale", "yshift", "death_rate", "mutation_rate"),
    Truth = c(1, 5, 1.5, 1, 1.3, 1.3)
)

priors <- list(
    xscale = \(x) dgamma(x, 2, scale = 1),
    xshift = \(x) dnorm(x, 5, 1),
    yscale = \(x) dgamma(x, 2, scale = 1),
    yshift = \(x) dnorm(x, 1, 1),
    death_rate = \(x) dlnorm(x, 0, 0.3),
    mutation_rate = \(x) dlnorm(x, 0, 0.5)
)

plot_hist <- function(parameter_name) {
    samples |>
        filter(Parameter == parameter_name) |>
        group_by(run, Parameter) |>
        summarise(Sample = median(Sample)) |>
        left_join(truth, by = "Parameter") |>
        ggplot() +
        geom_histogram(
            aes(x = Sample, y = after_stat(density), fill = "Sampling distribution")
        ) +
        geom_vline(aes(xintercept = Truth),
            color = "red",
            linewidth = 1
        ) +
        stat_function(
            aes(fill = "Prior"),
            geom = "area",
            fun = priors[[parameter_name]],
            alpha = 0.8
        ) +
        scale_fill_manual(
            name = "",
            values = c("Sampling distribution" = "steelblue", "Prior" = "grey")
        ) +
        facet_wrap(vars(Parameter)) +
        xlab("Estimate") +
        expand_limits(x = c(0, 4)) +
        theme_minimal() +
        theme(legend.position = "none")
}

if ("mutation_rate" %in% samples$Parameter) {
    (plot_hist("xscale") + plot_hist("xshift")) / (plot_hist("yscale") + plot_hist("yshift")) / (plot_hist("death_rate") + plot_hist("mutation_rate"))
} else {
    (plot_hist("xscale") + plot_hist("xshift")) / (plot_hist("yscale") + plot_hist("yshift")) / plot_hist("death_rate")
}

ggsave("plots.png", width = 10, height = 6, dpi = 300)
