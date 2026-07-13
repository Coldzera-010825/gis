# Correlation-network figures with linkET (R track) for Piece 09.
#
# linkET is the maintained successor to the (now-unmaintained) ggcor, written by
# the same author. It renders the same "correlation heatmap + Mantel links" figure
# and is compatible with modern ggplot2.
#
# Renders two PNGs into ../../figures:
#   corr-linket-heat.jpg    qcorrplot square/mark heatmap (linkET signature)
#   corr-linket-mantel.jpg  correlation heatmap + Mantel-test couples (the hero)
#
# Data: vegan's classic varespec (species) / varechem (soil chemistry).

suppressMessages({
  library(linkET)
  library(ggplot2)
  library(dplyr)
  library(vegan)
})

Sys.setenv(LANGUAGE = "en")
out_dir <- file.path("..", "..", "figures")   # run from visual/code/corrheatmap/
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

data("varespec")
data("varechem")

RdBu <- c("#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", "white",
          "#FDDBC7", "#F4A582", "#D6604D", "#B2182B")

# ---- Figure 1: qcorrplot heatmap of the soil-chemistry variables
p1 <- qcorrplot(correlate(varechem), type = "upper", diag = FALSE) +
  geom_square() +                                # square size + fill encode r
  geom_mark(sig_level = c(0.05, 0.01, 0.001),    # significance stars
            mark = c("*", "**", "***"), size = 2.6, colour = "grey15") +
  scale_fill_gradientn(colours = RdBu, limits = c(-1, 1),
                       name = "Pearson r") +
  labs(title = "linkET::qcorrplot  |  varechem soil chemistry")
ggsave(file.path(out_dir, "corr-linket-heat.jpg"), p1,
       width = 7.8, height = 6.6, dpi = 150)
cat("saved corr-linket-heat.jpg\n")

# ---- Figure 2: the Mantel network (correlation heatmap + linked species blocks)
# group the 44 species columns into four blocks, Mantel-test each against chemistry
mantel <- mantel_test(
  varespec, varechem,
  spec_select = list(Spec01 = 1:7, Spec02 = 8:18, Spec03 = 19:37, Spec04 = 38:44)
) %>%
  mutate(
    rd = cut(r, breaks = c(-Inf, 0.2, 0.4, Inf),
             labels = c("< 0.2", "0.2 - 0.4", ">= 0.4")),
    pd = cut(p, breaks = c(-Inf, 0.01, 0.05, Inf),
             labels = c("< 0.01", "0.01 - 0.05", ">= 0.05"))
  )

p2 <- qcorrplot(correlate(varechem), type = "upper", diag = FALSE) +
  geom_square() +
  geom_couple(aes(colour = pd, size = rd), data = mantel, curvature = 0.1) +
  scale_fill_gradientn(colours = RdBu, limits = c(-1, 1), name = "Pearson r") +
  scale_size_manual(values = c(0.5, 1.2, 2.2)) +
  scale_colour_manual(values = c("#D95F02", "#1B9E77", "grey70")) +
  guides(size = guide_legend(title = "Mantel r", order = 2),
         colour = guide_legend(title = "Mantel p", order = 1),
         fill = guide_colorbar(title = "Pearson r", order = 3)) +
  labs(title = "linkET Mantel network  |  species blocks vs soil chemistry")
ggsave(file.path(out_dir, "corr-linket-mantel.jpg"), p2,
       width = 9.8, height = 7.4, dpi = 150)
cat("saved corr-linket-mantel.jpg\n")
cat("done\n")
