"""Bounded, vectorised free-space evidence and static surface associations."""
import math

import cv2
import numpy as np


class FreeGrid:
    """Fixed memory timestamp grid, indexed by integer odom cells."""
    def __init__(self, resolution, half_size=4.):
        self.size = int(math.ceil(half_size*2/resolution))
        self.origin = np.array([-self.size//2, -self.size//2], dtype=int)
        self.data = np.full((self.size, self.size), -math.inf)

    def recenter(self, key):
        delta = np.asarray(key, dtype=int)-(self.origin+self.size//2)
        if np.max(np.abs(delta)) < self.size//4:
            return
        old = self.origin.copy()
        values = self.data
        self.origin = np.asarray(key, dtype=int)-self.size//2
        self.data = np.full_like(values, -math.inf)
        start = np.maximum(old, self.origin)
        end = np.minimum(old+self.size, self.origin+self.size)
        if np.any(end <= start):
            return
        a,b = start-old,end-old
        c,d = start-self.origin,end-self.origin
        self.data[c[0]:d[0],c[1]:d[1]] = values[a[0]:b[0],a[1]:b[1]]

    def get_many(self, keys):
        ij = np.asarray(keys, dtype=int)-self.origin
        mask = ((ij >= 0) & (ij < self.size)).all(axis=1)
        out = np.full(len(ij), -math.inf)
        out[mask] = self.data[ij[mask,0],ij[mask,1]]
        return out

    def set_many(self, keys, stamp):
        ij = np.asarray(keys, dtype=int).reshape(-1,2)-self.origin
        ij = ij[((ij >= 0) & (ij < self.size)).all(axis=1)]
        self.data[ij[:,0],ij[:,1]] = stamp

    def __setitem__(self, key, stamp):
        self.set_many([key], stamp)

    def get(self, key, default=-math.inf):
        v = self.get_many([key])[0]
        return default if not math.isfinite(v) else v

    def pop(self, key, default=None):
        old = self.get(key, default)
        self.set_many([key], -math.inf)
        return old

    def __len__(self):
        return int(np.isfinite(self.data).sum())

    def clear(self):
        self.data.fill(-math.inf)


class SurfaceTracks:
    """Track observed surfaces, not semantic vehicle counts or hidden box sizes."""
    def __init__(self):
        self.tracks = []
        self.next_id = 1
        self.revision = -1
        self._group_key = None
        self._groups = []

    def update(self, points, revision, now):
        if revision == self.revision:
            return self.tracks
        self.revision = revision
        if not len(points):
            self.tracks = []
            return self.tracks
        # Coarse adjacency joins sparse returns without expanding collision data.
        keys = np.floor(points/.075).astype(int)
        lo,hi = keys.min(axis=0),keys.max(axis=0)
        shape = hi-lo+1
        grid = np.zeros(tuple(shape), np.uint8)
        k = keys-lo
        grid[k[:,0],k[:,1]] = 1
        _, labels = cv2.connectedComponents(grid, connectivity=8)
        ids = labels[k[:,0],k[:,1]]
        tracks, used = [], set()
        for label in np.unique(ids):
            p = points[ids == label]
            centre = p.mean(axis=0)
            candidates = [(float(np.linalg.norm(centre-t['centre'])), i)
                          for i,t in enumerate(self.tracks) if i not in used]
            match = min(candidates, default=(math.inf,-1))
            if match[0] < .30:
                used.add(match[1])
                old = self.tracks[match[1]]
                ident, first = old['id'], old['first_seen']
            else:
                ident, first = self.next_id, now
                self.next_id += 1
            tracks.append(dict(id=ident, centre=centre, points=p, first_seen=first,
                               association_updated=now, partial=True))
        self.tracks = tracks
        return tracks

    def groups(self, distance=.50):
        """Nearby surfaces share a planning group, never a filled bounding box.

        Exact surface distance is used after a cheap bounding-box rejection.
        A gap within a group remains available to footprint collision checks.
        """
        key=(self.revision,float(distance))
        if key==self._group_key:
            return self._groups
        parent=list(range(len(self.tracks)))
        def root(i):
            while parent[i]!=i:
                parent[i]=parent[parent[i]];i=parent[i]
            return i
        bounds=[(t['points'].min(axis=0),t['points'].max(axis=0)) for t in self.tracks]
        for i,a in enumerate(self.tracks):
            for j in range(i):
                gap=np.maximum(0.,np.maximum(bounds[i][0]-bounds[j][1],bounds[j][0]-bounds[i][1]))
                if float(gap@gap)>distance*distance:
                    continue
                b=self.tracks[j]
                # Chunk pair distances to bound temporary memory.
                close=any(np.any(np.sum((p[:,None,:]-b['points'][None,:,:])**2,axis=2)<=distance**2)
                          for p in np.array_split(a['points'],max(1,math.ceil(len(a['points'])/64))))
                if close:
                    parent[root(i)]=root(j)
        members={}
        for i,t in enumerate(self.tracks):members.setdefault(root(i),[]).append(t)
        self._groups=[dict(id=min(t['id'] for t in ts),surface_ids=[t['id'] for t in ts],
                           points=np.concatenate([t['points'] for t in ts]),partial=True)
                      for ts in members.values()]
        self._group_key=key
        return self._groups
