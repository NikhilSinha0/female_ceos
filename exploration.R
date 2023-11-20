library(reticulate)
library(tidyverse)
py_install("pandas")
py_install("yfinance")
source_python('helpers/helpers.py')

create_graph <- function(name) {
    graph_data <- get_prices_by_ceo_name(name)
    ind_bef <- as.data.frame(graph_data[1]) |>
        mutate(val = "Industry Avg", status = "Before") |>
        rename(Value = industry_avg)

    ind_aft <- as.data.frame(graph_data[2]) |>
        mutate(val = "Industry Avg", status = "After") |>
        rename(Value = industry_avg)

    com_bef <- as.data.frame(graph_data[3]) |>
        mutate(val = "Company", status = "Before") |>
        rename(Value = Adj.Close)

    com_aft <- as.data.frame(graph_data[4]) |>
        mutate(val = "Company", status = "After") |>
        rename(Value = Adj.Close)

    ind_bef <- cbind(Day = rownames(ind_bef), ind_bef)
    rownames(ind_bef) <- 1:nrow(ind_bef)

    ind_aft <- cbind(Day = rownames(ind_aft), ind_aft)
    rownames(ind_aft) <- 1:nrow(ind_aft)

    com_bef <- cbind(Day = rownames(com_bef), com_bef)
    rownames(com_bef) <- 1:nrow(com_bef)

    com_aft <- cbind(Day = rownames(com_aft), com_aft)
    rownames(com_aft) <- 1:nrow(com_aft)

    df <- bind_rows(ind_bef, ind_aft, com_bef, com_aft)
    df$Day <- as.numeric(as.character(df$Day))

    ggplot(df, aes(x = Day, y = Value, color = val)) +
        facet_wrap(~factor(status, levels=c("Before", "After"))) +
        geom_line() +
        theme_bw()
}

create_graph("Mary T. Barra")
