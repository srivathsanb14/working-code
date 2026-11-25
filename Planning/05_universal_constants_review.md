# 05 · Review the universal constants once

**Phase:** Setup · **Priority:** P0 · **Size:** S · **Depends on:** nothing

## Goal

Every candidate formula may use the constants in `code/discovery/universal_constants.json` (they are listed in the
system prompt as `U[...]`). They must be values that are the same in every building and weather, each with a source.

## Steps

1. Read each entry against the kit code and `STATE_SPACE.md`: time step 300 s, cell 0.2 m, floor height 3 m, air
   volumetric heat capacity 1,206 J/m³K, 70 W per occupant, internal gains 10 W/m² with standby fraction 0.2, AHU and
   VAV flow limits, and the rest. Every entry already cites its kit source; confirm each value at that source.
2. Add any universal value a formula may need and is missing; never add a per-building value (wall or partition
   conductivity, density, specific heat, slab density, occupants per room, outside film): those must be inferred from
   the measured history.
3. Set `"_reviewed": true` (currently `false`) and note the reviewer and date in `_about`.

## Done when

- `_reviewed` is true, every value is confirmed, and the system prompt (`outputs/discovery/<run>/system_prompt.md`)
  shows the reviewed table.
