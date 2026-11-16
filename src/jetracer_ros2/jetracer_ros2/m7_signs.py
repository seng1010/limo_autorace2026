"""Bounded, acquisition-time A/B evidence; no ROS or detector dependency."""
from collections import deque
import math


class SignEvidence:
    def __init__(self):
        self.votes = deque(maxlen=5)
        self.last_stamp = -math.inf
        self.goal = None
        self.reason = 'waiting for a readable central view'

    def observe(self, detections, stamp, now, c, heading, config,
                image_height=480, pose_ready=True):
        if self.goal is not None:
            return None
        if (not isinstance(detections,(list,tuple)) or
                not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
                        for v in (stamp,now,c,heading,image_height))):
            self.reason = 'invalid sign metadata'
            return None
        if stamp<=self.last_stamp:
            return None
        self.last_stamp = stamp
        if not (-.035<=now-stamp<=.35) or image_height<=0:
            self.votes.clear();self.reason = 'sign image stale or invalid'
            return None
        if (not pose_ready or abs(c)>config.sign_center_half
                or abs(heading)>config.sign_heading_max):
            self.votes.clear();self.reason = 'sign viewpoint not central/aligned'
            return None
        aliases = {'a':'a','sign_a':'a','a_sign':'a','sign a':'a',
                   'b':'b','sign_b':'b','b_sign':'b','sign b':'b'}
        choices=[]
        for det in detections:
            if not isinstance(det,dict):
                continue
            label=aliases.get(str(det.get('label','')).lower().strip())
            try:
                box=[float(x) for x in det.get('bbox',())]
                confidence=float(det.get('confidence',0.))
            except (ValueError,TypeError):
                continue
            if (label is None or len(box)!=4 or not all(math.isfinite(x) for x in (*box,confidence))
                    or confidence<config.sign_min_confidence or box[2]<=box[0]
                    or (box[3]-box[1])*480./image_height<config.sign_min_height):
                continue
            choices.append((confidence,label))
        labels={label for _,label in choices}
        # Conflicting A/B boxes in one frame are not independent evidence.
        vote=next(iter(labels)) if len(labels)==1 else None
        self.votes=deque(((t,v) for t,v in self.votes if stamp-t<=.8),maxlen=5)
        self.votes.append((stamp,vote))
        for label,lane in (('a',0),('b',2)):
            times=[t for t,v in self.votes if v==label]
            if len(times)>=4 and times[-1]-times[0]>=.20:
                self.goal=lane;self.reason=f'{label.upper()} confirmed from central capture poses'
                return lane
        self.reason=f'central sign votes {sum(v is not None for _,v in self.votes)}/5'
        return None
