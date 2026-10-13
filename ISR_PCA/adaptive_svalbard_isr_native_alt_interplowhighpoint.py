#!/usr/bin/env python3
"""Build adaptive Svalbard ISR Ne products on consistent native altitude gates.

The highest altitude gate is retained for Svalbard, and profiles incompatible
with the year's dominant gate geometry are rejected before a rounded
altitude grid is built for the full year from gate ranks represented in at
least half of the year's profiles. Profiles with an altitude ceiling below 400 km are skipped.
The common grid is capped at 45 bins by default, decimating in native gate-rank
space across the full altitude range when needed. This preserves the native
nonuniform spacing pattern without allowing malformed profiles to create a
large, truncated low-altitude grid. The output altitude coordinate is a 1-D
native-style grid shared by all output profiles, not a uniform 10-km grid.
Incomplete time columns are removed after the single-missing-gate repair.
Below 500 km the default is a 4-minute median, switching to
10 minutes where the 10-minute mean deviation exceeds the threshold. Both the
10-minute adaptive median and the 30-minute topside median are centered moving
windows evaluated at every 4-minute output time. A topside boundary jump can be
repaired with a straight-line fit in log10(Ne) between the last point below
500 km and the highest valid point.

Example:
  python adaptive_svalbard_isr.py /path/to/SVL --year 2025 \
    --output-dir /path/to/adaptive_svl_native
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import netCDF4
import numpy as np

LOCAL_TZ = ZoneInfo("Europe/Oslo")
REQUIRED = {"timestamps", "gdalt", "ne"}

@dataclass
class Profile:
    timestamp: float
    altitude: np.ndarray
    ne: np.ndarray


def float_array(value):
    if np.ma.isMaskedArray(value):
        value = value.filled(np.nan)
    return np.asarray(value, dtype=float)


def read_profiles(path, year, alt_min, alt_max, ne_floor, ne_ceiling, chunk_profiles=32,
                  min_profile_top_km=400.0, drop_highest_gate=False):
    """Read native gates without imposing a common altitude grid."""
    result = {}
    with netCDF4.Dataset(path) as ds:
        missing = REQUIRED - set(ds.variables)
        if missing:
            raise ValueError("Missing variables: " + ", ".join(sorted(missing)))
        tv, zv, nv = (ds.variables[k] for k in ("timestamps", "gdalt", "ne"))
        times = float_array(tv[:]).reshape(-1)
        nt = times.size
        if tuple(zv.dimensions) != tuple(nv.dimensions):
            raise ValueError(f"gdalt/ne dimensions differ: {zv.dimensions} / {nv.dimensions}")
        dims = tuple(zv.dimensions)
        if dims[0] == "timestamps":
            time_axis = 0
        elif dims[-1] == "timestamps":
            time_axis = len(dims) - 1
        elif zv.shape[0] == nt:
            time_axis = 0
        elif zv.shape[-1] == nt:
            time_axis = len(dims) - 1
        else:
            raise ValueError(f"Cannot identify timestamps axis: {dims}, {zv.shape}")
        units = getattr(tv, "units", "Unix seconds")
        if " since " in units.lower():
            converted = netCDF4.num2date(times, units, only_use_cftime_datetimes=False,
                                         only_use_python_datetimes=True)
            times = np.array([x.replace(tzinfo=timezone.utc).timestamp()
                              if x.tzinfo is None else x.timestamp() for x in converted])
        elif "unix" not in units.lower() and "second" not in units.lower():
            raise ValueError(f"Unsupported timestamp units: {units}")

        for first in range(0, nt, chunk_profiles):
            stop = min(first + chunk_profiles, nt)
            if time_axis == 0:
                zchunk = float_array(zv[first:stop, :])
                nchunk = float_array(nv[first:stop, :])
            else:
                zchunk = float_array(zv[:, first:stop]).T
                nchunk = float_array(nv[:, first:stop]).T
            for k, stamp in enumerate(times[first:stop]):
                if not np.isfinite(stamp):
                    continue
                dt = datetime.fromtimestamp(float(stamp), timezone.utc)
                if dt.year != year:
                    continue
                z0, n0 = zchunk[k].reshape(-1), nchunk[k].reshape(-1)
                use_z = np.isfinite(z0) & (z0 >= alt_min) & (z0 <= alt_max)
                if np.count_nonzero(use_z) < 2:
                    continue
                z = z0[use_z]
                n = n0[use_z].copy()
                n[~(np.isfinite(n) & (n > ne_floor) & (n <= ne_ceiling))] = np.nan
                order = np.argsort(z)
                z, n = z[order], n[order]
                # Remove duplicate altitudes because interpolation requires an
                # ordered unique coordinate. Keep the first native gate.
                z_unique, unique_idx = np.unique(z, return_index=True)
                n = n[unique_idx]
                # Reject truncated or malformed profiles that only cover the
                # E region (for example, a 96-gate profile ending near 180 km).
                # Such profiles must not define the common daily altitude grid.
                if z_unique[-1] < min_profile_top_km:
                    continue
                # Svalbard commonly has only about 32 gates and its last gate
                # carries real coverage near 465 km, so retain it by default.
                if drop_highest_gate and z_unique.size > 2:
                    z_unique, n = z_unique[:-1], n[:-1]
                if np.count_nonzero(np.isfinite(n)) < 2:
                    continue
                result.setdefault(dt.strftime("%Y%m%d"), []).append(
                    Profile(float(stamp), z_unique.copy(), n.copy()))
    return result


def retain_dominant_geometry(profiles, gate_tolerance, bottom_tolerance_km,
                             top_tolerance_km):
    """Reject profile families incompatible with the dominant yearly mode."""
    if not profiles:
        return [], None
    counts=np.array([p.altitude.size for p in profiles],dtype=int)
    values,frequency=np.unique(counts,return_counts=True)
    reference_count=int(values[np.argmax(frequency)])
    near_count=np.abs(counts-reference_count)<=gate_tolerance
    if not np.any(near_count):
        return [],None
    bottoms=np.array([p.altitude[0] for p in profiles],dtype=float)
    tops=np.array([p.altitude[-1] for p in profiles],dtype=float)
    reference_bottom=float(np.median(bottoms[near_count]))
    reference_top=float(np.median(tops[near_count]))
    keep=(near_count &
          (np.abs(bottoms-reference_bottom)<=bottom_tolerance_km) &
          (np.abs(tops-reference_top)<=top_tolerance_km))
    summary=(reference_count,reference_bottom,reference_top)
    return [p for p,good in zip(profiles,keep) if good],summary


def adjacent_values(z, ne, target, adjacent_points, ne_floor, ne_ceiling):
    good = np.isfinite(z) & np.isfinite(ne) & (ne > ne_floor) & (ne <= ne_ceiling)
    if not np.any(good):
        return np.empty(0)
    zz, nn = z[good], ne[good]
    order = np.argsort(zz); zz, nn = zz[order], nn[order]
    if target < zz[0] or target > zz[-1]:
        return np.empty(0)
    center = int(np.argmin(np.abs(zz - target)))
    lo, hi = max(0, center-adjacent_points), min(len(zz), center+adjacent_points+1)
    return nn[lo:hi]


def interp_ne(src_z, src_ne, dst_z, max_gap_km):
    """Interpolate Ne linearly in log space without extrapolating or bridging large gaps."""
    out = np.full(np.shape(dst_z), np.nan, dtype=float)
    good = np.isfinite(src_z) & np.isfinite(src_ne) & (src_ne > 0)
    if np.count_nonzero(good) < 2:
        return out
    z, n = np.asarray(src_z)[good], np.asarray(src_ne)[good]
    order = np.argsort(z); z, n = z[order], n[order]
    z, idx = np.unique(z, return_index=True); n = n[idx]
    target = np.asarray(dst_z, dtype=float)
    inside = (target >= z[0]) & (target <= z[-1])
    if not np.any(inside):
        return out
    inds = np.searchsorted(z, target[inside], side="right")
    inds = np.clip(inds, 1, len(z)-1)
    left, right = inds-1, inds
    allowed = (z[right] - z[left]) <= max_gap_km
    vals = np.full(inds.shape, np.nan)
    if np.any(allowed):
        frac = (target[inside][allowed] - z[left[allowed]]) / (z[right[allowed]]-z[left[allowed]])
        vals[allowed] = 10 ** (np.log10(n[left[allowed]]) + frac *
                                (np.log10(n[right[allowed]])-np.log10(n[left[allowed]])))
    # Preserve exact source samples, including a singleton valid point.
    for i, (zz, nn) in enumerate(zip(z, n)):
        exact = np.isclose(target[inside], zz, atol=1e-6)
        vals[exact] = nn
    out[inside] = vals
    return out


def interp_scalar(src_z, src_y, dst_z, max_gap_km):
    good = np.isfinite(src_z) & np.isfinite(src_y)
    out = np.full(np.shape(dst_z), np.nan, dtype=float)
    if np.count_nonzero(good) < 2:
        return out
    z, y = np.asarray(src_z)[good], np.asarray(src_y)[good]
    order = np.argsort(z); z, y = z[order], y[order]
    z, idx = np.unique(z, return_index=True); y = y[idx]
    dst = np.asarray(dst_z, dtype=float)
    inside = (dst >= z[0]) & (dst <= z[-1])
    ii = np.searchsorted(z, dst[inside], side="right").clip(1, len(z)-1)
    lo, hi = ii-1, ii
    ok = (z[hi]-z[lo]) <= max_gap_km
    yy = np.full(ii.shape, np.nan)
    yy[ok] = y[lo[ok]] + (dst[inside][ok]-z[lo[ok]]) * (y[hi[ok]]-y[lo[ok]])/(z[hi[ok]]-z[lo[ok]])
    for zz, val in zip(z, y):
        yy[np.isclose(dst[inside], zz, atol=1e-6)] = val
    out[inside] = yy
    return out


def time_bin(stamp, day_start, width):
    return int((stamp-day_start.timestamp()) // (width*60))


def groups_by_interval(profiles, day_start, width):
    groups = [[] for _ in range(1440 // width)]
    for p in profiles:
        b = time_bin(p.timestamp, day_start, width)
        if 0 <= b < len(groups): groups[b].append(p)
    return groups


def common_daily_altitude(profiles, rounding_km, max_daily_gates=45,
                          min_gate_fraction=0.5):
    """Build a robust, rounded grid from gate ranks shared by most profiles.

    The native grid is nonuniform, so gate rank is retained. Gate ranks seen in
    fewer than ``min_gate_fraction`` of profiles are excluded. If the resulting
    consensus still has an unusually large number of gates, it is decimated
    evenly in gate-rank space to keep approximately the normal 40-gate scale
    while preserving the altitude span and native spacing pattern.
    """
    n_gates=max((p.altitude.size for p in profiles),default=0)
    if not n_gates: return np.array([],dtype=float)
    common=np.full(n_gates,np.nan)
    min_support=max(1,int(np.ceil(len(profiles)*min_gate_fraction)))
    for gate in range(n_gates):
        values=[p.altitude[gate] for p in profiles
                if p.altitude.size>gate and np.isfinite(p.altitude[gate])]
        if len(values)>=min_support: common[gate]=np.median(values)
    common=common[np.isfinite(common)]
    common=np.round(common/rounding_km)*rounding_km
    common=np.unique(common)
    if max_daily_gates > 0 and common.size > max_daily_gates:
        # Select across native gate rank, retaining both E-region and topside
        # coverage instead of truncating the upper end of the profile.
        keep=np.linspace(0,common.size-1,max_daily_gates).round().astype(int)
        common=common[np.unique(keep)]
    return common


def median_curve(group, args, common_grid):
    """Calculate median and sample standard deviation from identical samples."""
    if len(group) < args.min_profiles:
        return None
    zref=common_grid
    vals=np.full(zref.shape,np.nan)
    stds=np.full(zref.shape,np.nan)
    for ai,z in enumerate(zref):
        pieces=[adjacent_values(p.altitude,p.ne,z,args.adjacent_points,
                                args.ne_floor,args.ne_ceiling) for p in group]
        pieces=[x for x in pieces if x.size]
        if pieces:
            pooled=np.concatenate(pieces)
            if pooled.size>=args.min_samples:
                vals[ai]=np.median(pooled)
                stds[ai]=np.std(pooled,ddof=1)
    return zref.copy(),vals,stds


def median_curves(profiles, day_start, width, args, max_profiles, common_grid):
    """Median curves by fixed, non-overlapping time interval."""
    groups=groups_by_interval(profiles,day_start,width)
    curves=[None]*len(groups)
    for b,group in enumerate(groups):
        center=day_start.timestamp()+(b+0.5)*width*60
        if max_profiles is not None and len(group)>max_profiles:
            group=sorted(group,key=lambda p:abs(p.timestamp-center))[:max_profiles]
        curves[b]=median_curve(group,args,common_grid)
    return curves,groups


def moving_median_curves(profiles, day_start, args, common_grid, window_minutes,
                         max_profiles=None):
    """Centered rolling medians evaluated at every 4-minute output profile."""
    nout=1440//args.output_interval_minutes
    half_window=window_minutes*30.0
    curves=[None]*nout
    groups=[[] for _ in range(nout)]
    day0=day_start.timestamp()
    for j in range(nout):
        center=day0+(j+0.5)*args.output_interval_minutes*60
        group=[p for p in profiles if center-half_window<=p.timestamp<center+half_window]
        if max_profiles is not None and len(group)>max_profiles:
            group=sorted(group,key=lambda p:abs(p.timestamp-center))[:max_profiles]
        groups[j]=group
        curves[j]=median_curve(group,args,common_grid)
    return curves,groups


def choose_output_grids(profiles, day_start, args, n4, common_grid):
    # Same rounded native-gate grid at every output time during the day.
    return [common_grid.copy() for _ in range(n4)]


def median_to_grids(curves, width, grids, day_start, args):
    out = np.full((max((len(z) for z in grids), default=0), len(grids)), np.nan)
    for j, z in enumerate(grids):
        if not len(z): continue
        center_min = j*args.output_interval_minutes + args.output_interval_minutes/2
        b = min(int(center_min // width), len(curves)-1)
        curve = curves[b]
        if curve is None: continue
        vals = interp_ne(curve[0], curve[1], z, args.max_interp_gap_km)
        out[:len(z), j] = vals
    return out


def pad_grids(grids):
    nrow, ncol = max((len(z) for z in grids), default=0), len(grids)
    out = np.full((nrow, ncol), np.nan)
    for j,z in enumerate(grids): out[:len(z),j] = z
    return out


def deviation_decisions(profiles, short_curves, day_start, decision_grids, args):
    nrows=max((len(z) for z in decision_grids), default=0)
    means=np.full((nrows,len(decision_grids)),np.nan)
    counts=np.zeros((nrows,len(decision_grids)),dtype=np.int16)
    for b,zref in enumerate(decision_grids):
        if not len(zref): continue
        lo=day_start.timestamp()+b*args.decision_time_minutes*60
        hi=lo+args.decision_time_minutes*60
        raw=[p for p in profiles if lo <= p.timestamp < hi]
        accum=np.zeros(len(zref)); count=np.zeros(len(zref),dtype=int)
        for p in raw:
            sb=time_bin(p.timestamp,day_start,args.short_minutes)
            if sb<0 or sb>=len(short_curves) or short_curves[sb] is None: continue
            ref=interp_ne(short_curves[sb][0],short_curves[sb][1],zref,args.max_interp_gap_km)
            obs=interp_ne(p.altitude,p.ne,zref,args.max_interp_gap_km)
            valid=np.isfinite(ref)&(ref>0)&np.isfinite(obs)
            accum[valid]+=100*np.abs(obs[valid]-ref[valid])/ref[valid]
            count[valid]+=1
        good=count>=args.min_deviation_samples
        means[:len(zref),b][good]=accum[good]/count[good]
        counts[:len(zref),b]=count
    return means,counts


def repair_topside_jump(altitude, ne, boundary, ratio_threshold):
    """Repair only an abrupt boundary jump by a log10(Ne)-altitude line."""
    out=np.array(ne,dtype=float,copy=True)
    valid=np.isfinite(altitude)&np.isfinite(out)&(out>0)
    below=np.flatnonzero(valid&(altitude<boundary))
    above=np.flatnonzero(valid&(altitude>=boundary))
    if not below.size or not above.size: return out,False
    lo=below[np.argmax(altitude[below])]
    hi=above[np.argmin(altitude[above])]
    ratio=max(out[lo],out[hi])/min(out[lo],out[hi])
    if ratio < ratio_threshold: return out,False
    top=above[np.argmax(altitude[above])]
    if top==lo or altitude[top]<=altitude[lo]: return out,False
    z0,z1=altitude[lo],altitude[top]
    y0,y1=np.log10(out[lo]),np.log10(out[top])
    replace=np.isfinite(altitude)&(altitude>=boundary)
    out[replace]=10**(y0+(altitude[replace]-z0)*(y1-y0)/(z1-z0))
    return out,True


def construct_output(profiles, grids, short_curves, medium_curves, top_curves,
                     decision_mean, decision_grids, day_start, args):
    alt=pad_grids(grids)
    ne=np.full(alt.shape,np.nan)
    std=np.full(alt.shape,np.nan)
    duration=np.zeros(alt.shape,dtype=np.int8)
    repaired=np.zeros(len(grids),dtype=bool)
    for j,z in enumerate(grids):
        if not len(z): continue
        center_min=j*args.output_interval_minutes+args.output_interval_minutes/2
        ib4=min(int(center_min//args.short_minutes),len(short_curves)-1)
        # Both longer medians are centered on this 4-minute output profile.
        ibm=min(j,len(medium_curves)-1)
        ibt=min(j,len(top_curves)-1)
        short=short_curves[ib4]
        # Never allow a centered moving window to create an output column at a
        # time having no actual short-window Svalbard observations.
        if short is None:
            continue
        vals=interp_ne(short[0],short[1],z,args.max_interp_gap_km)
        spread=interp_scalar(short[0],short[2],z,args.max_interp_gap_km)
        ne[:len(z),j]=vals
        std[:len(z),j]=spread
        duration[:len(z),j]=args.short_minutes
        if decision_mean.shape[1]:
            idc=min(int(center_min//args.decision_time_minutes),decision_mean.shape[1]-1)
            zd=decision_grids[idc]
            dec=interp_scalar(zd,decision_mean[:len(zd),idc],z,args.max_interp_gap_km) if len(zd) else np.full(len(z),np.nan)
            use=np.isfinite(dec)&(dec>args.adaptive_threshold_pct)&(z<args.topside_boundary_km)
            curve=medium_curves[ibm]
            if curve is not None and np.any(use):
                vals=interp_ne(curve[0],curve[1],z,args.max_interp_gap_km)
                spread=interp_scalar(curve[0],curve[2],z,args.max_interp_gap_km)
                ne[:len(z),j][use]=vals[use]
                std[:len(z),j][use]=spread[use]
                duration[:len(z),j][use]=args.adaptive_minutes
        if z.size:
            use_top=z>=args.topside_boundary_km
            curve=top_curves[ibt]
            if curve is not None and np.any(use_top):
                vals=interp_ne(curve[0],curve[1],z,args.max_interp_gap_km)
                spread=interp_scalar(curve[0],curve[2],z,args.max_interp_gap_km)
                ne[:len(z),j][use_top]=vals[use_top]
                std[:len(z),j][use_top]=spread[use_top]
                duration[:len(z),j][use_top]=args.topside_minutes
        repaired_profile,was_repaired=repair_topside_jump(z,ne[:len(z),j],args.topside_boundary_km,args.topside_jump_ratio)
        ne[:len(z),j]=repaired_profile
        repaired[j]=was_repaired
    std[~np.isfinite(ne)]=np.nan
    return alt,ne,std,duration,repaired


def profile_deviation(profiles, altitude, median, args):
    out=np.full(median.shape,np.nan)
    for j in range(median.shape[1]):
        z=altitude[:,j]; okz=np.isfinite(z)
        if not np.any(okz): continue
        # Work in UTC-day seconds offset; profile time differences are assigned
        # to the same output interval as the median.
        daybase=(profiles[0].timestamp//86400)*86400 if profiles else 0
        lo=daybase+j*args.output_interval_minutes*60
        hi=lo+args.output_interval_minutes*60
        candidates=[p for p in profiles if lo<=p.timestamp<hi]
        if not candidates: continue
        bg=median[okz,j]; zz=z[okz]
        sums=np.zeros(len(zz)); counts=np.zeros(len(zz),dtype=int)
        for p in candidates:
            obs=interp_ne(p.altitude,p.ne,zz,args.max_interp_gap_km)
            valid=np.isfinite(obs)&np.isfinite(bg)&(bg>0)
            sums[valid]+=100*np.abs(obs[valid]-bg[valid])/bg[valid]
            counts[valid]+=1
        good=counts>0
        col=out[:,j]; inds=np.flatnonzero(okz)
        col[inds[good]]=sums[good]/counts[good]
        out[:,j]=col
    return out


def altitude_edges_2d(altitude):
    nrow,ncol=altitude.shape
    edges=np.full((nrow+1,ncol),np.nan)
    for j in range(ncol):
        z=altitude[:,j]; valid=np.flatnonzero(np.isfinite(z))
        if not valid.size: continue
        zz=z[valid]
        if len(zz)==1:
            edges[0,j]=zz[0]-1.5; edges[1,j]=zz[0]+1.5; continue
        mids=(zz[:-1]+zz[1:])/2
        e=np.r_[zz[0]-(zz[1]-zz[0])/2,mids,zz[-1]+(zz[-1]-zz[-2])/2]
        edges[:len(e),j]=e
    return edges


def interpolate_altitude_for_plot(values, altitude, log_scale=False,
                                  altitude_step_km=1.0, max_gap_km=60.0):
    """Interpolate only for display onto a fine vertical grid.

    Interpolation is limited to adjacent native gates that both have data, so
    missing gates stay blank. Electron density is interpolated in log space.
    This function never feeds values back into the NetCDF output.
    """
    zfinite=np.asarray(altitude,dtype=float)
    zfinite=zfinite[np.isfinite(zfinite)]
    if not zfinite.size:
        return np.array([],dtype=float),np.empty((0,values.shape[1]),dtype=float)
    zlo=np.floor(np.min(zfinite)/altitude_step_km)*altitude_step_km
    zhi=np.ceil(np.max(zfinite)/altitude_step_km)*altitude_step_km
    zplot=np.arange(zlo,zhi+altitude_step_km*0.5,altitude_step_km)
    out=np.full((zplot.size,values.shape[1]),np.nan,dtype=float)
    for j in range(values.shape[1]):
        zj=np.asarray(altitude[:,j],dtype=float)
        vj=np.asarray(values[:,j],dtype=float)
        valid_z=np.flatnonzero(np.isfinite(zj))
        if valid_z.size<2: continue
        order=np.argsort(zj[valid_z]); inds=valid_z[order]
        zz=zj[inds]; vv=vj[inds]
        for i in range(len(zz)-1):
            dz=zz[i+1]-zz[i]
            if (not np.isfinite(vv[i]) or not np.isfinite(vv[i+1]) or
                    dz<=0 or dz>max_gap_km):
                continue
            take=(zplot>=zz[i])&(zplot<=zz[i+1])
            if not np.any(take): continue
            fraction=(zplot[take]-zz[i])/dz
            if log_scale and vv[i]>0 and vv[i+1]>0:
                out[take,j]=10**(np.log10(vv[i])+fraction*(np.log10(vv[i+1])-np.log10(vv[i])))
            elif not log_scale:
                out[take,j]=vv[i]+fraction*(vv[i+1]-vv[i])
        # Preserve isolated native samples at the nearest displayed altitude.
        for zval,vval in zip(zz,vv):
            if np.isfinite(vval) and (not log_scale or vval>0):
                k=int(np.argmin(np.abs(zplot-zval)))
                if abs(zplot[k]-zval)<=altitude_step_km/2:
                    out[k,j]=vval
    return zplot,out


def plot_panel(values, altitude, times_utc, width_minutes, title, cbar_label,
               path, vmin, vmax, percent=False, log_scale=False):
    ncol=values.shape[1]
    centers=np.asarray(times_utc,dtype=float)
    x=mdates.epoch2num(centers) if hasattr(mdates,"epoch2num") else centers/86400+mdates.date2num(datetime(1970,1,1))
    fig,ax=plt.subplots(figsize=(14,7.5),constrained_layout=True)
    cmap=plt.colormaps["jet"].copy(); cmap.set_bad("white")
    norm=LogNorm(vmin=vmin,vmax=vmax) if log_scale else Normalize(vmin=vmin,vmax=vmax)
    zplot,shown=interpolate_altitude_for_plot(values,altitude,log_scale=log_scale)
    half_width=width_minutes/2880
    if ncol:
        xedges=np.r_[x[0]-half_width,(x[:-1]+x[1:])/2,x[-1]+half_width]
    else:
        xedges=np.array([],dtype=float)
    if zplot.size:
        yedges=np.r_[zplot[0]-0.5,zplot+0.5]
        mesh=ax.pcolormesh(xedges,yedges,shown,cmap=cmap,norm=norm,shading="flat",rasterized=True)
    else:
        mesh=ax.scatter([],[],c=[],cmap=cmap,norm=norm)
    cb=fig.colorbar(mesh,ax=ax,pad=.02,extend="max" if percent else "both"); cb.set_label(cbar_label)
    loc=mdates.AutoDateLocator(minticks=5,maxticks=10,tz=LOCAL_TZ)
    ax.xaxis.set_major_locator(loc); ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc,tz=LOCAL_TZ))
    if len(xedges): ax.set_xlim(xedges[0],xedges[-1])
    z=altitude[np.isfinite(altitude)]
    if z.size: ax.set_ylim(np.nanmin(z),np.nanmax(z))
    ax.set_xlabel("Local time (Europe/Oslo)"); ax.set_ylabel("Altitude (km; interpolated for display)"); ax.set_title(title)
    path.parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig)


def utc_time_strings(times):
    return np.array([datetime.fromtimestamp(float(t),timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
                     for t in times],dtype=object)


def local_time_strings(times):
    return np.array([datetime.fromtimestamp(float(t),timezone.utc).astimezone(LOCAL_TZ).strftime("%Y-%m-%dT%H:%M:%S")
                     for t in times],dtype=object)


def clean_complete_profiles(altitude, ne, durations, deviation, standard_deviation):
    """Keep complete profiles and repair profiles having exactly one bad gate.

    A gate is bad when any synchronized science variable is non-finite or its
    median interval is not positive.  For a single bad gate, all four values
    are copied from the closest fully valid gate in physical altitude.  A
    profile with two or more bad gates is rejected.
    """
    altitude=np.asarray(altitude,dtype=float)
    ne=np.array(ne,copy=True)
    durations=np.array(durations,copy=True)
    deviation=np.array(deviation,copy=True)
    standard_deviation=np.array(standard_deviation,copy=True)
    valid=(np.isfinite(ne) & np.isfinite(deviation) &
           np.isfinite(standard_deviation) & (durations>0))
    bad_count=np.sum(~valid,axis=0)
    keep=bad_count<=1
    repaired_single=0
    for j in np.flatnonzero(keep & (bad_count==1)):
        missing=int(np.flatnonzero(~valid[:,j])[0])
        sources=np.flatnonzero(valid[:,j] & np.isfinite(altitude))
        if sources.size==0 or not np.isfinite(altitude[missing]):
            keep[j]=False
            continue
        source=int(sources[np.argmin(np.abs(altitude[sources]-altitude[missing]))])
        ne[missing,j]=ne[source,j]
        durations[missing,j]=durations[source,j]
        deviation[missing,j]=deviation[source,j]
        standard_deviation[missing,j]=standard_deviation[source,j]
        repaired_single+=1
    complete=(np.all(np.isfinite(ne),axis=0) &
              np.all(np.isfinite(deviation),axis=0) &
              np.all(np.isfinite(standard_deviation),axis=0) &
              np.all(durations>0,axis=0))
    keep &= complete
    return keep,ne,durations,deviation,standard_deviation,repaired_single


def write_netcdf(path, date_text, altitude, times, ne, durations, deviation,
                 standard_deviation, repaired, args):
    expected_shape=(len(altitude),len(times))
    if not (ne.shape==durations.shape==deviation.shape==standard_deviation.shape==expected_shape):
        raise ValueError("Ne, deviation, standard deviation, and median interval arrays must share altitude x time shape")
    if len(repaired)!=len(times):
        raise ValueError("topside repair flags and all time coordinates must match the retained profile count")
    if not (np.all(np.isfinite(ne)) and np.all(np.isfinite(deviation)) and
            np.all(np.isfinite(standard_deviation)) and np.all(durations>0)):
        raise ValueError("Refusing to write incomplete retained profiles")
    path.parent.mkdir(parents=True,exist_ok=True)
    with netCDF4.Dataset(path,"w",format="NETCDF4") as ds:
        ds.title="Svalbard ISR adaptively median-filtered electron density on native altitude gates"
        ds.station_name=args.station_name
        ds.station_latitude=float(args.station_latitude)
        ds.station_longitude=float(args.station_longitude)
        ds.station_coordinate_units="degrees_north, degrees_east"
        ds.date_utc=date_text; ds.adaptive_threshold_percent=float(args.adaptive_threshold_pct)
        ds.topside_boundary_km=float(args.topside_boundary_km)
        ds.topside_jump_ratio_threshold=float(args.topside_jump_ratio)
        ds.median_policy=(f"{args.short_minutes}-min / centered moving {args.adaptive_minutes}-min below boundary; centered moving "
                          f"{args.topside_minutes}-min median above evaluated at every {args.output_interval_minutes}-min profile; "
                          "conditional topside jump repair")
        ds.time_grid_minutes=int(args.output_interval_minutes)
        ds.altitude_coordinate_note=("1-D altitude_gate coordinate shared by all time samples and days; formed from the median raw altitude at each native gate rank and rounded to "
                                     f"{args.altitude_rounding_km:g} km")
        ds.profile_completeness_policy=("profiles with two or more invalid altitude gates are removed; "
                                        "a profile with exactly one invalid gate is filled by copying Ne, "
                                        "deviation_percent, Ne_standard_deviation, and median_interval_minutes "
                                        "from the nearest fully valid altitude gate")
        ds.createDimension("altitude_gate",ne.shape[0]); ds.createDimension("time",len(times))
        z=ds.createVariable("altitude","f8",("altitude_gate",),zlib=True,complevel=3)
        z[:]=np.asarray(altitude,dtype=np.float64); z.units="km"; z.standard_name="altitude"; z.positive="up"
        t=ds.createVariable("time_utc",str,("time",)); t[:]=utc_time_strings(times)
        t.timezone="UTC"; t.description="UTC time formatted as YYYY-MM-DDTHH:MM:SS"
        lt=ds.createVariable("local_time",str,("time",)); lt[:]=local_time_strings(times)
        lt.timezone="Europe/Oslo"; lt.description="Local civil time formatted as YYYY-MM-DDTHH:MM:SS"
        te=ds.createVariable("time_utc_seconds","f8",("time",)); te[:]=times
        te.units="seconds since 1970-01-01 00:00:00 UTC"; te.standard_name="time"
        n=ds.createVariable("Ne","f4",("altitude_gate","time"),zlib=True,complevel=4,fill_value=np.float32(np.nan))
        n[:]=ne.astype(np.float32); n.units="m-3"; n.long_name="adaptively median-filtered electron density"
        d=ds.createVariable("median_interval_minutes","i1",("altitude_gate","time"),zlib=True,complevel=3)
        d[:]=durations; d.units="minutes"
        v=ds.createVariable("deviation_percent","f4",("altitude_gate","time"),zlib=True,complevel=4,fill_value=np.float32(np.nan))
        v[:]=deviation.astype(np.float32); v.units="%"
        s=ds.createVariable("Ne_standard_deviation","f4",("altitude_gate","time"),zlib=True,complevel=4,fill_value=np.float32(np.nan))
        s[:]=standard_deviation.astype(np.float32); s.units="m-3"
        s.long_name="sample standard deviation of the pooled raw electron-density samples used by the selected median"
        s.ddof=1; s.minimum_valid_profiles=int(args.min_profiles)
        s.minimum_pooled_samples=int(args.min_samples)
        s.valid_data_mask="identical to Ne"
        r=ds.createVariable("topside_jump_repaired","u1",("time",),zlib=True,complevel=3)
        r[:]=np.asarray(repaired,dtype=np.uint8); r.long_name="whether a sudden Ne jump at the topside boundary was replaced by a log-linear fit"


def process_day(day_text, profiles, args, figure_dir, daily_dir, common_grid,
                topside_profiles=None):
    day=datetime.strptime(day_text,"%Y%m%d").replace(tzinfo=timezone.utc)
    n4=1440//args.output_interval_minutes
    grids=choose_output_grids(profiles,day,args,n4,common_grid)
    short,_=median_curves(profiles,day,args.short_minutes,args,args.max_short_profiles,common_grid)
    top_source=profiles if topside_profiles is None else topside_profiles
    medium,_=moving_median_curves(
        top_source,day,args,common_grid,args.adaptive_minutes,
        args.max_adaptive_profiles)
    topside,_=moving_median_curves(
        top_source,day,args,common_grid,args.topside_minutes,
        args.max_topside_profiles)
    # Decision grids also use the shared year-level altitude coordinates.
    dec_groups=groups_by_interval(profiles,day,args.decision_time_minutes)
    dec_grids=[]
    for _ in dec_groups:
        dec_grids.append(common_grid.copy())
    dec_mean,dec_count=deviation_decisions(profiles,short,day,dec_grids,args)
    altitude,ne,std,duration,repaired=construct_output(
        profiles,grids,short,medium,topside,dec_mean,dec_grids,day,args)
    dev=profile_deviation(profiles,altitude,ne,args)
    times=day.timestamp()+(np.arange(n4)+.5)*args.output_interval_minutes*60
    xedges=[day+timedelta(minutes=i*args.output_interval_minutes) for i in range(n4+1)]
    slug=f"SVLISR{day_text}"
    plot_panel(ne,altitude,times,args.output_interval_minutes,
               f"Svalbard ISR adaptive Ne median ({args.short_minutes} min / centered {args.adaptive_minutes} min below {args.topside_boundary_km:g} km; centered {args.topside_minutes}-min above)",
               "Electron density Ne (m$^{-3}$)",figure_dir/f"{slug}_median.png",args.plot_ne_min,args.plot_ne_max,log_scale=True)
    plot_panel(dev,altitude,times,args.output_interval_minutes,
               "Deviation from selected median on native gates","Mean absolute relative deviation (%)",
               figure_dir/f"{slug}_deviation.png",0,args.deviation_plot_max,True)
    # Decision diagnostics shown at native altitude of each 10-minute reference profile.
    dec_times=np.array([day.timestamp()+(i+.5)*args.decision_time_minutes*60 for i in range(len(dec_grids))])
    dec_alt=np.full(dec_mean.shape,np.nan)
    for j,z in enumerate(dec_grids): dec_alt[:len(z),j]=z
    plot_panel(dec_mean,dec_alt,dec_times,args.decision_time_minutes,
               f"Mean deviation from {args.short_minutes}-min median; use {args.adaptive_minutes} min above {args.adaptive_threshold_pct:g}%",
               "Mean absolute relative deviation (%)",figure_dir/f"{slug}_adaptive_decision.png",0,args.deviation_plot_max,True)
    csv_path=figure_dir/f"{slug}_adaptive_decision.csv"
    with csv_path.open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["utc_start","local_start","altitude_km","valid_profiles","mean_deviation_from_short_pct","selected_interval_minutes"])
        for j,z in enumerate(dec_grids):
            utc=day+timedelta(minutes=j*args.decision_time_minutes); local=utc.astimezone(LOCAL_TZ)
            for i,alt in enumerate(z):
                value=dec_mean[i,j] if i<dec_mean.shape[0] else np.nan
                if np.isfinite(value): w.writerow([utc.isoformat(),local.isoformat(),alt,int(dec_count[i,j]),f"{value:.3f}",args.adaptive_minutes if value>args.adaptive_threshold_pct else args.short_minutes])
    keep,ne,duration,dev,std,nfilled=clean_complete_profiles(
        common_grid,ne,duration,dev,std)
    nc_path=daily_dir/f"{slug}.nc"
    if np.any(keep):
        write_netcdf(nc_path,day_text,common_grid,times[keep],ne[:,keep],duration[:,keep],
                     dev[:,keep],std[:,keep],repaired[keep],args)
    nrep=int(repaired.sum())
    print(f"  {day_text}: {len(profiles)} input profiles; kept {int(keep.sum())}/{n4} complete output columns; "
          f"single-gate fills {nfilled}; topside repairs {nrep}")
    return times,altitude,ne,duration,dev,std,keep,repaired


def parser_builder():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("input_path",type=Path); p.add_argument("--year",type=int,default=2025)
    p.add_argument("--recursive",action="store_true")
    p.add_argument("--output-dir",type=Path,default=Path("adaptive_svalbard_ne_native"))
    p.add_argument("--station-name",default="EISCAT Svalbard Radar")
    p.add_argument("--station-latitude",type=float,default=78.153)
    p.add_argument("--station-longitude",type=float,default=16.029)
    p.add_argument("--short-minutes",type=int,default=4); p.add_argument("--adaptive-minutes",type=int,default=10)
    p.add_argument("--topside-minutes",type=int,default=30)
    p.add_argument("--decision-time-minutes",type=int,default=10)
    p.add_argument("--adaptive-threshold-pct",type=float,default=30.0); p.add_argument("--min-deviation-samples",type=int,default=5)
    p.add_argument("--topside-boundary-km",type=float,default=500.0)
    p.add_argument("--topside-jump-ratio",type=float,default=1.5,help="Repair if adjacent Ne across boundary differs by this factor or more; set >1")
    p.add_argument("--alt-min",type=float,default=80.0); p.add_argument("--alt-max",type=float,default=700.0)
    p.add_argument("--altitude-rounding-km",type=float,default=0.1,
                   help="Round the per-day median native gate grid to this precision in km")
    p.add_argument("--max-daily-gates",type=int,default=45,
                   help="Maximum bins in the shared year-wide altitude grid; excess gates are decimated across the full altitude range")
    p.add_argument("--min-gate-fraction",type=float,default=0.5,
                   help="Require each native gate rank in this fraction of the year's profiles")
    p.add_argument("--min-profile-top-km",type=float,default=400.0,
                   help="Skip profiles whose highest usable altitude is below this value")
    p.add_argument("--drop-highest-gate",action="store_true",
                   help="Drop each raw profile's highest gate (off by default for Svalbard)")
    p.add_argument("--gate-count-tolerance",type=int,default=2,
                   help="Allowed gate-count difference from the dominant yearly profile family")
    p.add_argument("--profile-bottom-tolerance-km",type=float,default=20.0)
    p.add_argument("--profile-top-tolerance-km",type=float,default=60.0)
    p.add_argument("--adjacent-points",type=int,default=1); p.add_argument("--min-profiles",type=int,default=3)
    p.add_argument("--min-samples",type=int,default=6); p.add_argument("--max-short-profiles",type=int,default=4)
    p.add_argument("--max-adaptive-profiles",type=int,default=10); p.add_argument("--max-topside-profiles",type=int,default=30)
    p.add_argument("--max-interpolation-gap-km",dest="max_interp_gap_km",type=float,default=60.0)
    p.add_argument("--ne-floor",type=float,default=1e7); p.add_argument("--ne-ceiling",type=float,default=1e13)
    p.add_argument("--plot-ne-min",type=float,default=1e10); p.add_argument("--plot-ne-max",type=float,default=1e12)
    p.add_argument("--deviation-plot-max",type=float,default=100.0); p.add_argument("--chunk-profiles",type=int,default=32)
    return p


def main():
    args=parser_builder().parse_args()
    args.output_interval_minutes=4
    if args.topside_jump_ratio<=1: raise ValueError("--topside-jump-ratio must exceed 1")
    if args.max_interp_gap_km<=0 or args.altitude_rounding_km<=0: raise ValueError("Invalid interpolation gap or altitude rounding")
    for name in ("short_minutes","adaptive_minutes","topside_minutes","decision_time_minutes"):
        x=getattr(args,name)
        if x<=0 or 1440%x: raise ValueError(f"--{name.replace('_','-')} must divide 1440")
    if args.short_minutes>args.adaptive_minutes: raise ValueError("short interval must not exceed adaptive interval")
    if not 1<=args.min_deviation_samples<=args.decision_time_minutes: raise ValueError("invalid min deviation samples")
    if args.max_daily_gates < 2: raise ValueError("--max-daily-gates must be at least 2")
    if not 0 < args.min_gate_fraction <= 1: raise ValueError("--min-gate-fraction must be in (0, 1]")
    if args.gate_count_tolerance < 0: raise ValueError("--gate-count-tolerance cannot be negative")
    if args.profile_bottom_tolerance_km < 0 or args.profile_top_tolerance_km < 0:
        raise ValueError("Profile geometry tolerances cannot be negative")
    if args.input_path.is_file(): files=[args.input_path]
    elif args.input_path.is_dir(): files=sorted(args.input_path.glob("**/*.nc" if args.recursive else "*.nc"))
    else: raise FileNotFoundError(args.input_path)
    if not files: raise FileNotFoundError(f"No .nc files found under {args.input_path}")
    by_day={}; failures=[]
    for i,path in enumerate(files,1):
        print(f"[{i}/{len(files)}] Reading {path.name}",flush=True)
        try:
            got=read_profiles(path,args.year,args.alt_min,args.alt_max,args.ne_floor,args.ne_ceiling,
                               args.chunk_profiles,args.min_profile_top_km,args.drop_highest_gate)
            for day,ps in got.items(): by_day.setdefault(day,[]).extend(ps)
            print("    retained profiles:",sum(map(len,got.values())),flush=True)
        except Exception as exc:
            failures.append((str(path),f"{type(exc).__name__}: {exc}")); print(f"    SKIP: {type(exc).__name__}: {exc}",flush=True)
    if not by_day: raise RuntimeError(f"No usable profiles found for {args.year}")
    figure_dir=args.output_dir/"figures"; daily_dir=args.output_dir/"daily_netcdf"
    figure_dir.mkdir(parents=True,exist_ok=True); daily_dir.mkdir(parents=True,exist_ok=True)
    all_profiles=[p for daily_profiles in by_day.values() for p in daily_profiles]
    original_profile_count=len(all_profiles)
    all_profiles,geometry=retain_dominant_geometry(
        all_profiles,args.gate_count_tolerance,args.profile_bottom_tolerance_km,
        args.profile_top_tolerance_km)
    allowed={id(p) for p in all_profiles}
    for day in list(by_day):
        by_day[day]=[p for p in by_day[day] if id(p) in allowed]
        if not by_day[day]: del by_day[day]
    if not all_profiles:
        raise RuntimeError("No profiles matched the dominant Svalbard gate geometry")
    if geometry is not None:
        print(f"Dominant Svalbard geometry: {geometry[0]} gates, "
              f"{geometry[1]:.1f}-{geometry[2]:.1f} km; retained "
              f"{len(all_profiles)}/{original_profile_count} profiles")
    common_grid=common_daily_altitude(
        all_profiles,args.altitude_rounding_km,args.max_daily_gates,args.min_gate_fraction)
    if common_grid.size < 2:
        raise RuntimeError("Could not form a shared altitude grid with at least two bins")
    n_altitudes=common_grid.size
    del all_profiles
    year_start=datetime(args.year,1,1,tzinfo=timezone.utc)
    year_bins=(datetime(args.year+1,1,1,tzinfo=timezone.utc)-year_start).days*1440//args.output_interval_minutes
    shape=(n_altitudes,year_bins)
    annual_ne=np.full(shape,np.nan,dtype=np.float32)
    annual_dur=np.zeros(shape,dtype=np.int8); annual_dev=np.full(shape,np.nan,dtype=np.float32)
    annual_std=np.full(shape,np.nan,dtype=np.float32)
    annual_repaired=np.zeros(year_bins,dtype=np.uint8)
    processed=0
    previous_edge_profiles=[]
    for day_text in sorted(by_day):
        day_start=datetime.strptime(day_text,"%Y%m%d").replace(tzinfo=timezone.utc)
        day_end=day_start+timedelta(days=1)
        profiles=by_day[day_text]
        edge_seconds=max(args.topside_minutes,args.adaptive_minutes)*30
        previous=[p for p in previous_edge_profiles
                  if p.timestamp>=day_start.timestamp()-edge_seconds]
        next_day_text=day_end.strftime("%Y%m%d")
        next_profiles=[p for p in by_day.get(next_day_text,[])
                       if p.timestamp<day_end.timestamp()+edge_seconds]
        topside_profiles=previous+profiles+next_profiles
        times,alt,ne,dur,dev,std,keep,repaired=process_day(
            day_text,profiles,args,figure_dir,daily_dir,common_grid,topside_profiles)
        di=(datetime.strptime(day_text,"%Y%m%d").replace(tzinfo=timezone.utc)-year_start).days
        start=di*(1440//args.output_interval_minutes); stop=start+len(times)
        annual_ne[:,start:stop]=ne.astype(np.float32)
        annual_dur[:,start:stop]=dur; annual_dev[:,start:stop]=dev.astype(np.float32)
        annual_std[:,start:stop]=std.astype(np.float32)
        annual_repaired[start:stop]=repaired.astype(np.uint8)
        previous_edge_profiles=[p for p in profiles
                                if p.timestamp>=day_end.timestamp()-edge_seconds]
        processed+=1; del by_day[day_text]
    all_times=year_start.timestamp()+(np.arange(year_bins)+.5)*args.output_interval_minutes*60
    keep,annual_ne,annual_dur,annual_dev,annual_std,nfilled=clean_complete_profiles(
        common_grid,annual_ne,annual_dur,annual_dev,annual_std)
    annual_path=args.output_dir/f"SVLISR{args.year}.nc"
    write_netcdf(annual_path,str(args.year),common_grid,all_times[keep],annual_ne[:,keep],
                 annual_dur[:,keep],annual_dev[:,keep],annual_std[:,keep],
                 annual_repaired[keep],args)
    print(f"Saved annual native-altitude NetCDF: {annual_path} ({int(keep.sum())}/{year_bins} complete time columns; "
          f"single-gate fills {nfilled})")
    print(f"Processed {processed} UTC day(s); shared altitude grid has {n_altitudes} bins")
    if failures:
        with (args.output_dir/"failed_inputs.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.writer(f); w.writerow(["file","error"]); w.writerows(failures)
        print(f"Skipped {len(failures)} file(s); see {args.output_dir/'failed_inputs.csv'}")

if __name__=="__main__": main()
