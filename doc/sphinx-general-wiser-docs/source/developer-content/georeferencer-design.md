# Georeferencer Design
This document captures requirements and proposed design for the WISER Georeferencer tool.
## Problem Statement
Allow users to apply a spatial reference system and geo transform to an image by adding ground control points (GCPs) that map pixel coordinates in the target image to known geographic coordinates. This can be done using a reference image that already has spatial information, or by manually entering reference points.
## Scope
The tool is confined to its dialog window and its supporting classes:
- GeoreferencerPane: handles UI updates when the user clicks the target dataset to place GCPs.
- GeoReferenceTaskDelegate: handles the logic of adding GCPs when the user clicks between the target and reference image.
## Goals
- The user can add GCPs to the target image.
- GCPs can be added using either a reference image or manual entry.
## Background
WISER previously had no system to attach geographic information to datasets. The Georeferencer is the first step in doing so. It is particularly useful for hyperspWISER previously had no system to attach geographic information to datasets. The Georefrence system may exist.
## Functional Requirements
- Open any dataset currently loaded in WISER as the target image.
- Open any dataset with spatial information as the reference image.
- Handle out-of-memory datasets during georeferencing.
- Georeferencing computation must not block the main thread.
## Proposed Design
### High-Level Archite### Hi- GeoreferencerDialog: top-level dialog containing two GeoreferencerPane instances and the GeoReferencerTaskDelegate.
- GeoreferencerPane (x2): one for the target image, one for the reference image. Updates display when user clicks to place a GCP.
- GeoReferencerTaskDelegate: coordinates GCP placem- GeoReferencerTaskDelegate: coordinatesa Model
To be documented as implementation proceeds.
### UI/UX
Mockups toMockups toM