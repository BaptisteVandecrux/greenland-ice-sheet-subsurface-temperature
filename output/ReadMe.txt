This dataset contains the subsurface temperature observation interpolated at a 10 m depth compiled for the following study:

contact: bav@geus.dk
Vandecrux, B., Fausto, R. S., Box, J., Covi, F., Hock, R., Rennermalm, A., Heilig, A., Abermann, J., van As, D., Bjerre, E., Fettweis, X., Smeets, P.C.J.P., Kuipers Munneke, P., van den Broeke, M., Brils, M., Langen, P.L., Mottram, R., Ahlstrøm, A.: Historical snow and ice temperature observations document the recent warming of the Greenland ice sheet, Submitted to the Cryosphere, 2023.

When using this dataset please cite both the study above and the individual studies listed in the "reference" field.

Data processing:

 For the temperatures continuously recorded by thermistor or thermocouple strings, the depth of each temperature sensor below the surface was calculated using installation depths and recorded surface height. Wherever necessary, we interpolated the available temperature profiles linearly to 10 m depth and allowed linear extrapolation if at least two measurements were available within 2 m of the 10 m depth. The resulting T10m values were then aggregated as monthly means if they originated from continuous measurements, or left as instantaneous values otherwise. 

Data structure:

In the 10m_temperature_dataset_monthly.csv file, each data point has the following attributes:

    date: in YYY-MM-DD format
    site
    latitude
    longitude
    elevation
    depthOfTemperatureObservation: currently 10 m for all. May expend to all depths in future releases
    temperatureObserved
    reference: see list below
    reference_short
    note
    error
    durationOpen (for compatibility with the SUMup dataset often not known)
    durationMeasured: Indicate if a measurement is a monthly average or an instantaneous value
    method

Dataset composition:

Reference						Start     End        Number of measurements
Koch (1913)                    	1912    1913    5
Wegener (1930)                	1930    1930    8
Heuberger (1954)         	   	1950    1950    2
Benson (1962)                	1954    1955    59
Schytt (1955)                	1954    1954    31
Nobles (1960)                	1954    1954    7
Heuberger (1954)            	1954    1954    1
Meier et al. (1957)            	1955    1955    4
Griffiths (1960)            	1955    1956    38
de Quervain (1969)            	1957    1964    8
Ambach (1979)                	1959    1959    2
Langway (1961)                	1959    1959    14
U.S. Army Transportation
Board (1960)                	1960    1960    4
Davies (1954)                	1960    1960    7
Davies (1967)                	1962    1962    1
Mock (1965)                    	1964    1964    12
Mock and Ragle (1963)        	1964    1964    31
Weertman et al. (1968)        	1966    1966    1
Colbeck and Gow (1979)        	1973    1973    3
Clausen et al. (1988)        	1974    1985    11
Clausen and Hammer (1988)    	1977    1977    1
Stauffer and Oeschger
(1979)                        	1978    1978    3
Clement (1984)                	1983    1983    4
Thomsen et al. (1991)        	1990    1991    8
Ohmura et al. (1992)        	1990    1990    3
GC-Net unpublished            	1991    2010    170
Braithwaite (1993)            	1991    1992    12
Laternser (1994)            	1992    1992    16
Schwager (2000)                	1994    1994    1
Historical GC-Net:            	1995    2022    1665
    Steffen et al. (1996, 2001, 2023);
    Vandecrux et al. (2023)
Giese and Hawley (2015)        	2004    2008    47
Humphrey et al. (2012)        	2007    2009    57
PROMICE:                    	2008    2023    1346
    Fausto et al. (2021);
    How et al. (2022)        
Smeets et al. (2018)        	2009    2016    160
Harrington et al. (2015)    	2010    2012    5
Hills et al. (2018)            	2011    2017    109
Charalampidis et al. (2016) &
Charalampidis et al. (2022)    	2012    2013    29
Yamaguchi et al. (2014)        	2012    2012    1
Miller et al. (2020)        	2013    2017    68
Polashenski et al. (2014)    	2013    2013    2
Matoba et al. (2015)        	2014    2014    1
MacFerrin et al.            	2015    2019    311
(2021, 2022)
Kjær et al. (2015)            	2015    2015    2
Heilig et al. (2018)        	2016    2021    58
Vandecrux et al. (2021);
Colgan and Vandecrux (2021)    	2017    2022    119
Covi et al. (2022, 2023)    	2017    2019    77
Law et al. (2021)            	2019    2019    1
GC-Net continuation:        	2021    2023    134
Fausto et al. (2021); How et al. (2022)

Total:    4659


References:

Ambach, W., Zum Wärmehaushalt des Grönländischen Inlandeises: Vergleichende Studie im Akkumulations- und Ablationsgebiet,  Polarforschung 49 (1): 44-54, 1979

Benson, C. S. (1962)  Stratigraphic  studies  in the snow and firn of  the  Greenland ice sheet, U. S.  Army  Snow Ice  and  Permafrost  Research  Establishment (USA SIPRE) Research Report 70, 93p

Braithwaite, R. (1993). Firn temperature and meltwater refreezing in the lower accumulation area of the Greenland ice sheet, Pâkitsoq, West Greenland. Rapport Grønlands Geologiske Undersøgelse, 159, 109–114. https://doi.org/10.34194/rapggu.v159.8218

Charalampidis, C., Van As, D., Colgan, W.T., Fausto, R.S., Macferrin, M. and Machguth, H., 2016. Thermal tracing of retained meltwater in the lower accumulation area of the Southwestern Greenland ice sheet. Annals of Glaciology, 57(72), pp.1-10.

Clausen HB and Stauffer B (1988) Analyses of Two Ice Cores Drilled at the Ice-Sheet Margin in West Greenland. Annals of Glaciology 10, 23–27 (doi:10.3189/S0260305500004109)

Clausen, H., N. Gundestrup, S. Johnsen, R. Bindschadler and J. Zwally (1988), Glaciological investigations in the Crete area, Central Greenland: A search for a new deep-drilling site. Ann. Glaciol.,10, 10-15.

Clausen, H., and Hammer, C. (1988). The laki and tambora eruptions as revealed in Greenland ice cores from 11 locations. J. Glaciology 10, 16–22. doi:10.1017/s026030550000409

Clement, P. “Glaciological Activities in the Johan Dahl Land Area, South Greenland, As a Basis for Mapping Hydropower Potential”. Rapport Grønlands Geologiske Undersøgelse, vol. 120, Dec. 1984, pp. 113-21, doi:10.34194/rapggu.v120.7870.

Colbeck S. and A. Gow. 1979. The margin of the Greenland Ice Sheet at Isua. Journal of Glaciology. 24: 155-165. 10.3189/S0022143000014714

Covi, F., Hock, R., and Reijmer, C.: Challenges in modeling the energy balance and melt in the percolation zone of the Greenland ice sheet. Journal of Glaciology, 69(273), 164-178. doi:10.1017/jog.2022.54, 2023

Covi, F., Hock, R., Rennermalm, A., Leidman S., Miege, C., Kingslake, J., Xiao, J., MacFerrin, M., Tedesco, M.: Meteorological and firn temperature data from three weather stations in the percolation zone of southwest Greenland, 2017 - 2019. Arctic Data Center. doi:10.18739/A2BN9X444, 2022.

Davies, T.C., Structures in the upper snow layers of the southern Dome Greenland ice sheet, CRREL research report 115, 1954

Davis RM: Approach roads Greenland 1960?1964  Technical Report 133. Corps of Engineers Cold Regions Research & Engineering Laboratory  1967

Fausto, R. S., van As, D., Mankoff, K. D., Vandecrux, B., Citterio, M., Ahlstrøm, A. P., Andersen, S. B., Colgan, W., Karlsson, N. B., Kjeldsen, K. K., Korsgaard, N. J., Larsen, S. H., Nielsen, S., Pedersen, A. Ø., Shields, C. L., Solgaard, A. M., and Box, J. E.: Programme for Monitoring of the Greenland Ice Sheet (PROMICE) automatic weather station data, Earth Syst. Sci. Data, 13, 3819–3845, https://doi.org/10.5194/essd-13-3819-2021 , 2021. and How, P., Ahlstrøm, A.P., Andersen, S.B., Box, J.E., Citterio, M., Colgan, W.T., Fausto, R., Karlsson, N.B., Jakobsen, J., Larsen, S.H., Mankoff, K.D., Pedersen, A.Ø., Rutishauser, A., Shields, C.L., Solgaard, A.M., van As, D., Vandecrux, B., Wright, P.J., PROMICE and GC-Net automated weather station data in Greenland, https://doi.org/10.22008/FK2/IW73UU, GEUS Dataverse, 2022.

Giese AL and Hawley RL (2015) Reconstructing thermal properties of firn at Summit, Greenland, from a temperature profile time series. Journal of Glaciology 61(227), 503–510 (doi:10.3189/2015JoG14J204)

Griffiths, T. M. (1960). Glaciological investigations in the TUTO area of Greenland., U. S. Army Snow Ice and Permafrost Research Establishment, Corps of Engineers, Report 47, 62 pp. https://hdl.handle.net/11681/5981

Harrington Joel A. Humphrey Neil F. Harper Joel T.: Temperature distribution and thermal anomalies along a flowline of the Greenland ice sheet  Annals of Glaciology 56(70) 98?104 2015 10.3189/2015AoG70A945

Heilig, A., Eisen, O., MacFerrin, M., Tedesco, M., and Fettweis, X.: Seasonal monitoring of melt and accumulation within the deep percolation zone of the Greenland Ice Sheet and comparison with simulations of regional climate modeling, The Cryosphere, 12, 1851–1866, https://doi.org/10.5194/tc-12-1851-2018, 2018.

Heuberger J.-C. 1954. Expéditions Polaires Françaises: Missions Paul-Emil Victor. Glaciologie Groenland Volume 1: Forages sur l'inlandsis. Hermann & Cle Éditeurs. Paris.

Heuberger, Jean-Charles (1954) Groenland, glaciologie, Vol. I, Forages sur L'inlandsis (Greenland, glaciology, vol. I, Borehole studies on the ice cap). Paris: Hermann & Cle, Editeurs.

Hills, B. H., Harper, J. T., Meierbachtol, T. W., Johnson, J. V., Humphrey, N. F., and Wright, P. J.: Processes influencing heat transfer in the near-surface ice of Greenlands ablation zone, The Cryosphere, 12, 3215–3227, https://doi.org/10.5194/tc-12-3215-2018, 2018. data: https://doi.org/10.18739/A2QV3C418

Humphrey, N. F., Harper, J. T., and Pfeffer, W. T. (2012), Thermal tracking of meltwater retention in Greenlands accumulation area, J. Geophys. Res., 117, F01010, doi:10.1029/2011JF002083. Data available at: https://instaar.colorado.edu/research/publications/occasional-papers/firn-stratigraphy-and-temperature-to-10-m-depth-in-the-percolation-zone-of/

Kjær, H. A., Zens, P., Edwards, R., Olesen, M., Mottram, R., Lewis, G., Terkelsen Holme, C., Black, S., Holst Lund, K., Schmidt, M., Dahl-Jensen, D., Vinther, B., Svensson, A., Karlsson, N., Box, J. E., Kipfstuhl, S., and Vallelonga, P.: Recent North Greenland temperature warming and accumulation, The Cryosphere Discuss. [preprint], https://doi.org/10.5194/tc-2020-337 , 2021.

Koch, Johann P., and Alfred Wegener. Wissenschaftliche Ergebnisse Der Dänischen Expedition Nach Dronning Louises-Land Und Quer über Das Inlandeis Von Nordgrönland 1912 - 13 Unter Leitung Von Hauptmann J. P. Koch : 1 (1930). 1930.

Langway,  C. C.,  Jr. (1961) Accumulation and temperature on the inland ice of North Greenland, 1959, Journal of Glaciology,  vol.  3, no.  30, p.  1017-1044.

Laternser, M., 1994 Firn temperature measurements and snow pit studies on the EGIG traverse of central Greenland, 1992. Eidgenössische Technische Hochschule.  Versuchsanstalt für Wasserbau  Hydrologie und Glaziologic. (Arbeitsheft 15).

Law, R., Christoffersen, P., Hubbard, B., Doyle, S.H., Chudley, T.R., Schoonman, C.M., Bougamont, M., des Tombe, B., Schilperoort, B., Kechavarzi, C. and Booth, A., 2021. Thermodynamics of a fast-moving Greenlandic outlet glacier revealed by fiber-optic distributed temperature sensing. Science Advances, 7(20), p.eabe7136. DOI: 10.1126/sciadv.abe713

MacFerrin, M. J., Stevens, C. M., Vandecrux, B., Waddington, E. D., and Abdalati, W. (2022) The Greenland Firn Compaction Verification and Reconnaissance (FirnCover) dataset, 2013–2019, Earth Syst. Sci. Data, 14, 955–971, https://doi.org/10.5194/essd-14-955-2022,

Matoba, S., Motoyama, H., Fujita, K., Yamasaki, T., Minowa, M., Onuma, Y., Komuro, Y., Aoki, T., Yamaguchi, S., Sugiyama, S., Enomoto, H., 2015. Glaciological and meteorological observations at the SIGMA-D site, northwestern Greenland Ice Sheet. Bull. Glaciol. Res. 33, 7–14. https://doi.org/10.5331/bgr.33.7

Meier, M. F., Conel, J. E., Hoerni, J. A., Melbourne, W. G., & Pings, C. J. (1957). Preliminary Study of Crevasse Formation. Blue Ice Valley, Greenland, 1955. OCCIDENTAL COLL LOS ANGELES CALIF. https://hdl.handle.net/11681/6029

Miller, O., Solomon, D.K., Miège, C., Koenig, L., Forster, R., Schmerr, N., Ligtenberg, S.R., Legchenko, A., Voss, C.I., Montgomery, L. and McConnell, J.R., 2020. Hydrology of a perennial firn aquifer in Southeast Greenland: an overview driven by field data. Water Resources Research, 56(8), p.e2019WR026348. Dataset doi:10.18739/A2R785P5W

Mock, S.  J. (1965)  Glaciological  studies  in the vicinity  of Camp Century, Green­ land, U. S. Army Cold Regions Research and Engineering Laboratory  (USA CRREL) Research Report 157.

Rinker, J. N. and Mock, S. J. (in preparation) Radar ice sounding data, Green­ land 1964, USA CRREL Special Report.

Mock, S. J. and Ragle, R. H.  ( 1963} Elevations on the ice sheet of southern Greenland USA CRREL Technical Report 124, 9p. Ragle, R. H. and Davis, T. C. (1962) South Greenland traverses, Journal of Glaciology, vol. 4, p. 129-131.

Nobles, L. H., Glaciological investigations, Nunatarssuaq ice ramp, Northwestern Greenland, Tech. Rep. 66, U.S. Army Snow, Ice and Permafrost Research Establishment, Corps of Engineers, 1960.

Ohmura, A. and 10 others. 1992; Energy and Mass balance during the melt season at the equilibrium line altitude, Paakitsoq, Greenland ice sheet. Zürich, Swiss Federal Institute of Technology. (ETH Greenland Expedition. Progress Report 2.)

Polashenski, C., Z. Courville, C. Benson, A. Wagner, J. Chen, G. Wong, R. Hawley, and D. Hall (2014), Observations of pronounced Greenland ice sheet firn warming and implications for runoff production, Geophys. Res. Lett., 41, 4238–4246, doi:10.1002/2014GL059806.

de Quervain, M. (1969), Schneckundliche Arbeiten der Internat. Glaziolog. Gronlandexpedition. Meddelelser om Gronland, Bd. 177, Nr. 4.

Schwager, M. (2000): Eisbohrkernuntersuchungen zur räumlichen und zeitlichen Variabilität von Temperatur und Niederschlagsrate im Spätholozän in Nordgrönland - Ice core analysis on the spatial and temporal variability of temperature and precipitation during the late Holocene in North Greenland , Berichte zur Polarforschung (Reports on Polar Research), Bremerhaven, Alfred Wegener Institute for Polar and Marine Research, 362 , 136 p. . doi: 10.2312/BzP_0362_2000

Schytt, V. (1955) Glaciological investigations in the Thule Ramp area, U. S. Army Snow Ice and Permafrost Research Establishment, Corps of Engineers, Report 28, 88 pp. https://hdl.handle.net/11681/5989

Stauffer B. and H. Oeschger. 1979. Temperaturprofile in bohrloechern am rande des Groenlaendischen Inlandeises. Hydrologie und Glaziologie an der ETH Zurich. Mitteilung Nr. 41.

Steffen, K., Box, J.E. and Abdalati, W., 1996. Greenland climate network: GC-Net. US Army Cold Regions Reattach and Engineering (CRREL), CRREL Special Report, pp.98-103.

Steffen, K. and J. Box:  Surface climatology of the Greenland ice sheet: Greenland Climate Network 1995-1999, J. Geophys. Res., 106, 33,951-33,972, 2001

Steffen, K., Vandecrux, B., Houtz, D., Abdalati, W., Bayou, N., Box, J., Colgan, L., Espona Pernas, L., Griessinger, N., Haas-Artho, D., Heilig, A., Hubert, A., Iosifescu Enescu, I., Johnson-Amin, N., Karlsson, N. B., Kurup, R., McGrath, D., Cullen, N. J., Naderpour, R., Pederson, A. Ø., Perren, B., Philipps, T., Plattner, G.K., Proksch, M., Revheim, M. K., Særrelse, M., Schneebli, M., Sampson, K., Starkweather, S., Steffen, S., Stroeve, J., Watler, B., Winton, Ø. A., Zwally, J., Ahlstrøm, A.: GC-Net Level 1 automated weather station data, https://doi.org/10.22008/FK2/VVXGUT, GEUS Dataverse, V2, 2023.

Smeets, P.C. J. P., Peter Kuipers Munneke, Dirk van As, Michiel R. van den Broeke, Wim Boot, Hans Oerlemans, Henk Snellen, Carleen H. Reijmer & Roderik S. W. van de Wal (2018) The K-transect in west Greenland: Automatic weather station data (1993–2016), Arctic, Antarctic, and Alpine Research, 50:1, DOI: 10.1080/15230430.2017.1420954

Sorge, E. Glaziologische Untersuchungen in Eismitte, 5. Beitrag. p62-263 in K. Wegener: Wissenschaftliche Ergebnisse der deutschen Grönland-Expedition Alfred Wegener 1929 und 1930/1931 Bd. III Glaziologie.

Thomsen H.H. O.B. Olesen R.J. Braithwaite and C.E. Bøggild. 1991. Ice drilling and mass balance at Pakitsoq Jakobshavn central West Greenland. Rapport Grï¿½nlands Geologiske Undersï¿½gelse 152 80?84. 10.34194/rapggu.v152.8160

Thomsen, H. ., Olesen, O. ., Braithwaite, R. . and Bøggild, C. .: Ice drilling and mass balance at Pâkitsoq, Jakobshavn, central West Greenland, Rapp. Grønlands Geol. Undersøgelse, 152, 80–84, doi:10.34194/rapggu.v152.8160, 1991.

U.S. Army Transportation Board (1960) Report of environmental operation: Lead Dog 1960, Final Report, Project Lead Dog, TCB- 60 - 023 - E0, 188p. https://apps.dtic.mil/sti/pdfs/AD0263548.pdf

Vandecrux, B., Colgan, W., Solgaard, A.M., Steffensen, J.P., and Karlsson, N.B.(2021). Firn evolution at Camp Century, Greenland: 1966-2100, Frontiers in Earth Science, https://doi.org/10.3389/feart.2021.578978, 2021 dataset: https://doi.org/10.22008/FK2/SR3O4F

Vandecrux, B., Box, J.E., Ahlstrøm, A.P., Andersen, S.B., Bayou, N., Colgan, W.T., Cullen, N.J., Fausto, R.S., Haas-Artho, D., Heilig, A., Houtz, D.A., How, P., Iosifescu Enescu , I., Karlsson, N.B., Kurup Buchholz, R., Mankoff, K.D., McGrath, D., Molotch, N.P., Perren, B., Revheim, M.K., Rutishauser, A., Sampson, K., Schneebeli, M., Starkweather, S., Steffen, S., Weber, J., Wright, P.J., Zwally, J., Steffen, K.: The historical Greenland Climate Network (GC-Net) curated and augmented Level 1 dataset, Submitted to ESSD, 2023

Weertman J.: Comparison between measured and theoretical temperature profiles of the Camp Century Greenland Borehole  Journal of Geophysical Research 73(8) American Geophysical Union (AGU) 2691?2700 4 1968 10.1029/jb073i008p02691

Wegener, K.: Wissenschaftliche Ergebnisse der deutschen Grönland-Expedition Alfred Wegener 1929 und 1930/1931 Bd. III Glaziologie.

Yamaguchi, S., Matoba, S., Yamazaki, T., Tsushima, A., Niwano, M., Tanikawa, T., Aoki, T., 2014. Glaciological observations in 2012 and 2013 at SIGMA-A site, Northwest Greenland. Bull. Glaciol. Res. 32, 95–105. https://doi.org/10.5331/bgr.32.95


