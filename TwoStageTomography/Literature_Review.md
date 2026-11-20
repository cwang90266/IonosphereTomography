# E-Region Retrieval from RO TEC — Literature Notes

Companion to `E_Region_Retrieval_Plan.md`. Compiled 2026-11-01 from web
searches. Most entries were checked at abstract level only; full texts
were not read. Before quoting numbers in a report, check them against the
original papers.

## 1. Why Abel inversion fails in the E region

| Reference | Finding | Use for us |
| --- | --- | --- |
| Kelley, Wong, Aponte, Coker, Mannucci, Komjathy (2009), *Radio Sci.* 44, doi:10.1029/2008RS004087 — COSMIC vs Arecibo ISR | A simple Abel transform is **not suitable for E-region profiles, even in daytime**. NmF2 reasonable, hmF2 biased. | The bar to beat: Abel E-region is a weak baseline. |
| Yue et al. (2010), "Error analysis of Abel retrieved electron density profiles from radio occultation measurements", *Ann. Geophys.* 28, 217 ([pdf](https://angeo.copernicus.org/articles/28/217/2010/angeo-28-217-2010.pdf)) | Errors from the spherical-symmetry assumption propagate **downward** from the top and can give negative E-region densities. Largest at low latitudes (EIA). | Same downward aliasing happens in our joint EnKF through E–F covariance. |
| Shaikh, Notarpietro, Nava (2014), *Adv. Space Res.* ([link](https://www.sciencedirect.com/science/article/abs/pii/S0273117713006595)) | Assessment of the spherical-symmetry impact on RO inversion. | Background. |
| Dietrich et al. (2024), "Evaluating RO constellation designs using OSSEs for ionospheric specification", *Space Weather* ([link](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024SW003958)) | Densities below ~200 km are an inherent limitation of RO/Abel. Says spherical-symmetry error still needs ray-path quantification. | Supports doing our own OSSE (WP0) before investing. |
| Hysell (2007), "Inverting ionospheric RO measurements using maximum entropy", *Radio Sci.* ([link](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2007RS003635)) | Regularized (MaxEnt) inversion as an alternative to onion peeling. | Regularization idea for the stage-2 E solve. |

## 2. Aided inversions: get F from elsewhere, then solve for E

| Reference | Finding | Use for us |
| --- | --- | --- |
| Hernández-Pajares, Juan, Sanz (2000), *GRL* 27(16) — "Improving the Abel inversion by adding ground GPS data to LEO RO" | **Separability**: Ne = VTEC(lat, lon) × shape(h). Gradients from ground VTEC. Good enough to measure **E-layer critical frequency**. | Closest precedent to "fix the horizontal/F structure, then retrieve E". |
| Garcia-Fernandez et al. (2003, 2005 *GPS Solutions* 9:105) and Aragón-Ángel et al. (2009/2010, *GPS Solutions* 14:23) ([UPC pdf](https://server.gage.upc.edu/papers/2006/fulltext-10.pdf)) | Separability applied to COSMIC, including bending angles. Better foF2 than Abel and an average **E-layer improvement** at different latitudes. A later check found separability does not hold for all shape functions. | Expect gains but not uniformly. Validate per latitude and time. |
| Wu et al. (2019), "Evaluation of Abel inversion method assisted by an improved IRI model", *JGR Space Phys.* ([link](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019JA026880)) | Spherical symmetry is the dominant error source and gives large E-region errors. An IRI-assisted Abel reduces them. | Model-aided version of our stage 2. |
| Yue et al. (2012), "A feasibility study of the RO electron density retrieval aided by a global ionospheric data assimilation model", *JGR* ([link](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2011JA017446)) | The horizontal structure is taken from a DA analysis and then used in the RO inversion. | Our stage 1 plays exactly this role for stage 2. |
| GIM-aided Abel, *GPS Solutions* (2012) ([link](https://link.springer.com/article/10.1007/s10291-012-0281-9)) and background-model-assisted Abel, *JASTP* (2014) ([link](https://www.sciencedirect.com/science/article/abs/pii/S1364682614002909)) | Helps the E and F regions **if the aiding map is accurate**. | The F-error propagation (member pairing, §6.1 of the plan) is needed because the aiding field is never exact. |
| Gradient-aided inversion, *Adv. Space Res.* (2010) ([link](https://www.sciencedirect.com/science/article/abs/pii/S027311771000459X)) | Drops the spherical assumption. Better E and F accuracy in simulations. | Background. |

## 3. Using the shape of the low TEC profile and high-rate data

| Reference | Finding | Use for us |
| --- | --- | --- |
| Wu, D. L. (2022), "Global GNSS-RO electron density in the lower ionosphere", *Remote Sensing* ([Zenodo](https://zenodo.org/record/6057919)). See also Wu (2018, *JASTP*), "New global electron density observations from GPS-RO in the D- and E-region ionosphere" | **Removes the F-region contribution in excess phase with a linear fit** before retrieving D/E-region Ne from high-rate RO (the high-rate top is at least 120 km). | Direct support for the differenced/curvature observable (WP2) and for using high-rate conPhs data (WP1). |
| Wu et al. (2023), Remote Sensing Part I: optimal-estimation retrieval of Ne from GNSS-POD hTEC at negative elevation ([DOAJ](https://doaj.org/article/2b546426f89a4b7883022b2d1ebdf141); [NASA pdf](https://ntrs.nasa.gov/api/citations/20230009528/downloads/20230009528%20Wu%2C%20D%20paper.pdf)) | OE and onion peeling both use the Abel weighting function but differ significantly in the lower ionosphere (~150 km). hTEC and its vertical derivative are strongly related. Navigation error in hTEC is 1–3 m (6–18 TECU). Suggests tomography using asymmetric sampling about the tangent point. | Error-budget reference: absolute errors are large and smooth, so use relative/derivative information. Supports our tomographic (non-Abel) approach. |
| GFZ report (Arras 2010) ([pdf](https://gfzpublic.gfz-potsdam.de/pubman/item/item_23022_8/component/file_23021/1009.pdf)) | Open-loop sampling at **50 Hz** in the lower part gives about **50 m** tangent-height resolution. | Sets what WP1 high-rate data could reach. |

## 4. Sporadic E

| Reference | Finding | Use for us |
| --- | --- | --- |
| Arras et al. (2008), first global Es climatology from CHAMP, GRACE and COSMIC. EGU 2014 abstract ([pdf](https://meetingorganizer.copernicus.org/EGU2014/EGU2014-5813.pdf)) | Es appears as strong **L1 SNR fluctuations** above ~85 km. | Es flag from the podTc2 SNR (WP5). |
| Yue et al. (2015), *AMT* 8, 225 ([pdf](https://amt.copernicus.org/articles/8/225/2015/amt-8-225-2015.pdf)) | During Es, phase and SNR fluctuate together. SNR has a U-shaped profile. | Use both the TEC/phase and SNR signatures. |
| Carmona et al. (2022), *Remote Sensing* 14, 581 ([pdf](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-14-00581/article_deploy/remotesensing-14-00581.pdf)) | Compares 5 Es methods (Chu, Arras: SNR std > 0.2; Yu; Niu: TEC-perturbation gradient > 0.12 TECU/km; Gooch: fbEs > 3 MHz) against ionosondes, for tangents of 70–130 km. | Ready-made thresholds for the WP5 flags. |
| Sobhkhiz-Miandehi et al. (2023), *Front. Astron. Space Sci.* ([link](https://www.frontiersin.org/journals/astronomy-and-space-sciences/articles/10.3389/fspas.2023.1198071/epub)) | At least 20 % of Es detections disagree between ionosonde and RO. | Do not over-trust Es flags. Treat them as an error-inflation trigger, not truth. |
| 3-D tomographic reconstruction of Es layers with GNSS-RO constraints, *J. Geodesy* (2025) ([link](https://link.springer.com/article/10.1007/s00190-025-01962-6)) | Ground-GNSS tomography constrained by RO traces the Es layer altitude. | A future extension: a thin-layer basis element (WP5 option c). |

## 5. Tomography and assimilation frameworks

| Reference | Finding | Use for us |
| --- | --- | --- |
| Lin, Matsuo, Liu et al. (2015, *AMT* 8, 171) ([link](https://amt.copernicus.org/articles/8/171/2015/)); Lin et al. (2017, *JGR*) "Data assimilation of ground-based GPS and RO TEC for global ionospheric specification" | Kalman filter on IRI with **EOF covariances from an IRI ensemble**. Location-dependent covariance outperforms location-independent. Validated with Millstone Hill ISR; F2 focus. | Nearest analogue to our PCA styles. They report F-peak, not E, results, which matches our F success and E difficulty. |
| Two-step tomography: Tikhonov + MART (Wen et al. 2012); NeQuick2 prior + MART (Jin & Li 2018) ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0273117718304460)) | Step 1 fixes large-scale structure; step 2 refines it. | Precedent for staged solutions, though aimed at the F region. |
| Parameterized tomography, *GPS Solutions* (2019) ([link](https://link.springer.com/article/10.1007/s10291-019-0893-4)) | Voxel tomography is ill-posed. Vertical structure is better handled by parameterization; best at 25 km vertical resolution for ground data. | Refining the grid alone is not enough; the prior/basis must support it (WP3 + WP4 together). |
| IONOLAB-Fusion (2024), *Atmosphere* 15, 675 ([doi](https://doi.org/10.3390/atmos15060675)) | Adding RO to ground tomography is what provides vertical resolution. | Future: IGS ground TEC + RO jointly (already supported by `igs_source`). |

**Gap:** I found no published RO tomography that does F first, then E on a
refined E-region grid with F uncertainty propagated. The two-stage design
in the plan is a new combination of established pieces (aided inversion,
F-background removal, block priors, ensemble DA).

## 6. Validation resources

| Reference | Finding | Use for us |
| --- | --- | --- |
| Stolle et al. (2004), *Ann. Geophys.* 22, 2015 ([link](https://angeo.copernicus.org/articles/22/2015/2004/)) | CHAMP RO vs **EISCAT** (28 events). Most peak and TEC differences lie within the combined errors, and they grow when the ionosphere is structured. | Same radar family as our TRO/SVL validation. Expect noisy comparisons in structured conditions. |
| Salinas et al. (2024), E-PROBED model, *Space Weather* ([UCAR](https://impacts.ucar.edu/en/publications/development-of-the-ionospheric-e-region-prompt-radio-occultation-/)) | Empirical COSMIC-1 E-region (90–120 km) model. Good vs ISR/ionosonde by day, **large disagreement at night**. | An independent daytime E climatology for checking the prior (WP4). Night-time E is the hardest case. |
| Bender et al. (2021), SSUSI vs EISCAT auroral E region, *Ann. Geophys.* 39, 899 ([link](https://angeo.copernicus.org/articles/39/899/2021/)) | Auroral E-region Ne from UV imagery agrees with EISCAT to ~0–20 % above 100–105 km. | Possible independent auroral E data source at TRO. |
