import xarray
from matplotlib.pyplot import figure


def timeprofile(iono: xarray.Dataset):
    fig = figure(figsize=(16, 12))
    axs = fig.subplots(3, 1, sharex=True).ravel()

    fig.suptitle(
        f"{str(iono.time[0].values)[:-13]} to "
        f"{str(iono.time[-1].values)[:-13]}\n"
        f"Glat, Glon: {iono.glat.item()}, {iono.glon.item()}"
    )

    ax = axs[0]
    ax.plot(iono.time, iono["NmF2"], label="N$_m$F$_2$")
    ax.plot(iono.time, iono["NmF1"], label="N$_m$F$_1$")
    ax.plot(iono.time, iono["NmE"], label="N$_m$E")
    ax.set_title("Maximum number densities vs. ionospheric layer")
    ax.set_ylabel("(m$^{-3}$)")
    ax.set_yscale("log")
    ax.legend(loc="best")
    ax = axs[1]
    ax.plot(iono.time, iono["hmF2"], label="h$_m$F$_2$")
    ax.plot(iono.time, iono["hmF1"], label="h$_m$F$_1$")
    ax.plot(iono.time, iono["hmE"], label="h$_m$E")
    ax.set_title("Height of maximum density vs. ionospheric layer")
    ax.set_ylabel("(km)")
    ax.set_ylim((90, None))
    ax.legend(loc="best")
    ax = axs[2]
    ax.plot(iono.time, iono["foF2"], label="foF2")
    ax.set_title("F2 layer plasma frequency")
    ax.set_ylabel("(MHz)")

    for a in axs.ravel():
        a.grid(True)

    # %%
    fig = figure(figsize=(16, 12))
    axs = fig.subplots(1, 1, sharex=True)

    fig.suptitle(
        f"{str(iono.time[0].values)[:-13]} to "
        f"{str(iono.time[-1].values)[:-13]}\n"
        f"Glat, Glon: {iono.glat.item()}, {iono.glon.item()}"
    )
    # %% Tec(time)
    ax = axs
    ax.plot(iono.time, iono["TEC"], label="TEC")
    ax.set_ylabel("(m$^{-2}$)")
    ax.set_title("Total Electron Content (TEC)")
    # ax.set_yscale('log')
    ax.legend(loc="best")
    ax.grid(True)
    # %% ion_drift(time)
    # ax = axs[1]
    # ax.plot(iono.time, iono["EqVertIonDrift"], label=r"V$_y$")
    # ax.set_xlabel("time (UTC)")
    # ax.set_ylabel("(m/s)")
    # ax.legend(loc="best")

    # for a in axs.ravel():
    #    a.grid(True)

    # %%  Ne(time)
    fg = figure()
    ax = fg.gca()
    hi = ax.pcolormesh(iono.time, iono.alt_km, iono["ne"].values.T)
    fg.colorbar(hi, ax=ax).set_label("[m$^{-3}$]")
    ax.set_ylabel("altitude [km]")
    ax.set_title("$N_e$ vs. altitude and time")


def altprofile(iono: xarray.Dataset):
    fig = figure(figsize=(16, 6))
    axs = fig.subplots(1, 2)

    fig.suptitle(
        f"{str(iono.time[0].values)[:-13]}\n" f"Glat, Glon: {iono.glat.item()}, {iono.glon.item()}"
    )

    pn = axs[0]
    pn.plot(iono["ne"], iono.alt_km, label="N$_e$")

    pn.set_xlabel("Density (m$^{-3}$)")
    pn.set_ylabel("Altitude (km)")
    pn.set_xscale("log")
    pn.legend(loc="best")
    pn.grid(True)

    pn = axs[1]
    pn.plot(iono["Ti"], iono.alt_km, label="T$_i$")
    pn.plot(iono["Te"], iono.alt_km, label="T$_e$")

    pn.set_xlabel("Temperature (K)")
    pn.set_ylabel("Altitude (km)")
    pn.legend(loc="best")
    pn.grid(True)

def altprofile_sensitivity(iParam_Switch:int,iono_initial: xarray.Dataset, 
    iono_foF2: [xarray.Dataset],iono_hmF2: [xarray.Dataset],
    iono_B0: [xarray.Dataset],iono_B1: [xarray.Dataset]):
    fig = figure(figsize=(16, 8))
    axs = fig.subplots(2, 2)
    if iParam_Switch==1:
        fig.suptitle(
            f"{str(iono_initial.time[0].values)[:-13]}\n" 
            f"Glat, Glon: {iono_initial.glat.item()},{iono_initial.glon.item()}"
            )
    else:
        fig.suptitle(
            f"{str(iono_initial.time[0].values)[:-13]}\n" 
            f"Glat, Glon: {iono_initial.glat.item()},{iono_initial.glon.item()}"
            )
            
    
    pn = axs[0,0]
    for iono in iono_foF2:
        pn.plot(iono["ne"], iono.alt_km, color=(0.8,0.8,0.8))

    if iParam_Switch==1:
        pn.plot(iono_initial["ne"], iono_initial.alt_km,linewidth=3,color=(0,0,0),
                label=f"Nominal, foF2: {iono_initial.foF2.item()}")
        iono = iono_foF2[0]
        pn.plot(iono["ne"], iono.alt_km,color=(1,0,0), label=f"min foF2: {iono.foF2.item()}")
        iono = iono_foF2[-1]
        pn.plot(iono["ne"], iono.alt_km,color=(0,0,1), label=f"max foF2: {iono.foF2.item()}")
    else:
        pn.plot(iono_initial["ne"], iono_initial.alt_km,linewidth=3,color=(0,0,0),
                label=f"Nominal, f07D: {iono_initial.f107D.item()}")
        iono = iono_foF2[0]
        pn.plot(iono["ne"], iono.alt_km,color=(1,0,0), label=f"min f107D: {iono.f107D.item()}")
        iono = iono_foF2[1]
        pn.plot(iono["ne"], iono.alt_km,color=(0,1,0), label=f"mean f107D: {iono.f107D.item()}")
        iono = iono_foF2[2]
        pn.plot(iono["ne"], iono.alt_km,color=(1,0,1), label=f"median f107D: {iono.f107D.item()}")
        iono = iono_foF2[3]
        pn.plot(iono["ne"], iono.alt_km,color=(0,0,1), label=f"max f107D: {iono.f107D.item()}")
        
    pn.set_xlabel("Density (m$^{-3}$)")
    pn.set_ylabel("Altitude (km)")
    pn.set_xscale("log")
    pn.legend(loc="best")
    pn.grid(True)

    pn = axs[0,1]
    for iono in iono_hmF2:
        pn.plot(iono["ne"], iono.alt_km, color=(0.8,0.8,0.8))

    if iParam_Switch==1:
        pn.plot(iono_initial["ne"], iono_initial.alt_km,linewidth=3,color=(0,0,0),
             label=f"Nominal, hmF2: {iono_initial.hmF2.item()}")
        iono = iono_hmF2[0]
        pn.plot(iono["ne"], iono.alt_km,color=(1,0,0), label=f"min hmF2: {iono.hmF2.item()}")
        iono = iono_hmF2[-1]
        pn.plot(iono["ne"], iono.alt_km,color=(0,0,1), label=f"max hmF2: {iono.hmF2.item()}")
    else:
        pn.plot(iono_initial["ne"], iono_initial.alt_km,linewidth=3,color=(0,0,0),
             label=f"Nominal, ap: {iono_initial.ap.item()}")
        iono = iono_hmF2[0]
        pn.plot(iono["ne"], iono.alt_km,color=(1,0,0), label=f"min ap: {iono.ap.item()}")
        iono = iono_hmF2[1]
        pn.plot(iono["ne"], iono.alt_km,color=(0,1,0), label=f"mean ap: {iono.ap.item()}")
        iono = iono_hmF2[2]
        pn.plot(iono["ne"], iono.alt_km,color=(1,0,1), label=f"median ap: {iono.ap.item()}")
        iono = iono_hmF2[3]
        pn.plot(iono["ne"], iono.alt_km,color=(0,0,1), label=f"max ap: {iono.ap.item()}")
    pn.set_xlabel("Density (m$^{-3}$)")
    pn.set_ylabel("Altitude (km)")
    pn.set_xscale("log")
    pn.legend(loc="best")
    pn.grid(True)

    pn = axs[1,0]
    for iono in iono_B0:
        pn.plot(iono["ne"], iono.alt_km, color=(0.8,0.8,0.8))

    if iParam_Switch==1:
        pn.plot(iono_initial["ne"], iono_initial.alt_km,linewidth=3,color=(0,0,0),
             label=f"Nominal, B0: {iono_initial.B0.item()}")
        iono = iono_B0[0]
        pn.plot(iono["ne"], iono.alt_km,color=(1,0,0), label=f"min B0: {iono.B0.item()}")
        iono = iono_B0[-1]
        pn.plot(iono["ne"], iono.alt_km,color=(0,0,1), label=f"max B0: {iono.B0.item()}")
    else:
        pn.plot(iono_initial["ne"], iono_initial.alt_km,linewidth=3,color=(0,0,0),
             label=f"Nominal, IG12: {iono_initial.IG12.item()}")
        iono = iono_B0[0]
        pn.plot(iono["ne"], iono.alt_km,color=(1,0,0), label=f"min IG12: {iono.IG12.item()}")
        iono = iono_B0[1]
        pn.plot(iono["ne"], iono.alt_km,color=(0,1,0), label=f"mean IG12: {iono.IG12.item()}")
        iono = iono_B0[2]
        pn.plot(iono["ne"], iono.alt_km,color=(1,0,1), label=f"median IG12: {iono.IG12.item()}")
        iono = iono_B0[-1]
        pn.plot(iono["ne"], iono.alt_km,color=(0,0,1), label=f"max IG12: {iono.IG12.item()}")
    pn.set_xlabel("Density (m$^{-3}$)")
    pn.set_ylabel("Altitude (km)")
    pn.set_xscale("log")
    pn.legend(loc="best")
    pn.grid(True)

    pn = axs[1,1]
    for iono in iono_B1:
        pn.plot(iono["ne"], iono.alt_km, color=(0.8,0.8,0.8))

    if iParam_Switch==1:
        pn.plot(iono_initial["ne"], iono_initial.alt_km,linewidth=3,color=(0,0,0),
             label=f"Nominal, B1: {iono_initial.B1.item()}")
        iono = iono_B1[0]
        pn.plot(iono["ne"], iono.alt_km,color=(1,0,0), label=f"min B1: {iono.B1.item()}")
        iono = iono_B1[-1]
        pn.plot(iono["ne"], iono.alt_km,color=(0,0,1), label=f"max B1: {iono.B1.item()}")
    else:
        pn.plot(iono_initial["ne"], iono_initial.alt_km,linewidth=3,color=(0,0,0),
             label=f"Nominal, Rz12: {iono_initial.Rz12.item()}")
        iono = iono_B1[0]
        pn.plot(iono["ne"], iono.alt_km,color=(1,0,0), label=f"min Rz12: {iono.Rz12.item()}")
        iono = iono_B1[1]
        pn.plot(iono["ne"], iono.alt_km,color=(0,1,0), label=f"mean Rz12: {iono.Rz12.item()}")
        iono = iono_B1[2]
        pn.plot(iono["ne"], iono.alt_km,color=(1,0,1), label=f"median Rz12: {iono.Rz12.item()}")
        iono = iono_B1[3]
        pn.plot(iono["ne"], iono.alt_km,color=(0,0,1), label=f"max Rz12: {iono.Rz12.item()}")
    pn.set_xlabel("Density (m$^{-3}$)")
    pn.set_ylabel("Altitude (km)")
    pn.set_xscale("log")
    pn.legend(loc="best")
    pn.grid(True)
   
def latprofile(iono: xarray.Dataset):
    fig = figure(figsize=(8, 12))
    axs = fig.subplots(2, 1, sharex=True)

    ax = axs[0]

    ax.plot(iono["glat"], iono["NmF2"], label="N$_m$F$_2$")
    ax.plot(iono["glat"], iono["NmF1"], label="N$_m$F$_1$")
    ax.plot(iono["glat"], iono["NmE"], label="N$_m$E")
    ax.set_title(str(iono.time[0].values)[:-13] + f'  latitude {iono["glat"][[0, -1]].values}')
    # ax.set_xlim(iono.lat[[0, -1]])
    ax.set_xlabel(r"Geog. Lat. ($^\circ$)")
    ax.set_ylabel("(m$^{-3}$)")
    ax.set_yscale("log")

    ax = axs[1]
    ax.plot(iono["glat"], iono["hmF2"], label="h$_m$F$_2$")
    ax.plot(iono["glat"], iono["hmF1"], label="h$_m$F$_1$")
    ax.plot(iono["glat"], iono["hmE"], label="h$_m$E")
    ax.set_xlim(iono["glat"][[0, -1]])
    ax.set_title(str(iono.time[0].values)[:-13] + f'  latitude  {iono["glat"][[0, -1]].values}')
    ax.set_xlabel(r"Geog. Lat. ($^\circ$)")
    ax.set_ylabel("(km)")

    for a in axs:
        a.legend(loc="best")
        a.grid(True)
