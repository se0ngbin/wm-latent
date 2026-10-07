"""Dreamer analog of critic_ood_eval.py: sign-acc / AUC of the reachability
critic's V vs HJ V* under appearance shift. Feat = observe->get_feat (544-d)."""
import sys, argparse
from pathlib import Path
import numpy as np, torch
_THIS = Path(__file__).resolve()
for p in [str(_THIS.parents[3]), str(_THIS.parents[2]), str(_THIS.parents[1]), str(_THIS.parent)]:
    if p not in sys.path: sys.path.append(p)
import gym as old_gym  # noqa: F401
import gymnasium as gym  # noqa: F401
from PIL import Image, ImageDraw
from sklearn.metrics import roc_auc_score
from configs import DreamerConfig, Config
from latent_cbf.dubins.dubins_env import DubinsEnv
from compare_critic_to_hj_dreamer import build_dreamer_wm
from critic_ood_eval import hj_interp, load_ddpg, eval_V, render_diamond, CONDS, HJ, DEV, N

from PyHJ.utils.net.common import Net  # noqa: F401  (load_ddpg imported from critic_ood_eval)


def make_env():
    ec = Config().environment
    return DubinsEnv(image_size=tuple(ec.image_size), world_bounds=tuple(ec.world_bounds),
                     max_angular_velocity=ec.max_angular_velocity, speed=ec.speed, dt=ec.dt,
                     obstacles=ec.get_obstacles_list(), goal_radius=ec.goal_radius,
                     collision_radius=ec.collision_radius, render_mode="rgb_array")


def render_cond(env, st, cond):
    env.colors = {"background":"white","agent":"blue","goal":"green","obstacle":cond["obstacle"]}
    dia = cond["shape"]=="diamond"; rot = cond.get("rot",0); imgs=[]
    for s in st:
        env.state=s.astype(np.float32)
        im=render_diamond(env) if dia else env.render()
        if rot: im=np.ascontiguousarray(np.rot90(im,k=rot//90))
        imgs.append(im)
    return np.stack(imgs,0)


@torch.no_grad()
def feats(wm, imgs, ths, bs=128):
    out=[]
    for i in range(0,imgs.shape[0],bs):
        ch,tc=imgs[i:i+bs],ths[i:i+bs]; b=ch.shape[0]
        obs_state=np.stack([np.cos(tc),np.sin(tc)],-1)[:,None].astype(np.float32)
        batch={"image":ch[:,None],"obs_state":obs_state,"action":np.zeros((b,1,1),np.float32),
               "is_first":np.ones((b,1,1),np.float32),"is_terminal":np.zeros((b,1,1),np.float32)}
        data=wm.preprocess(batch); embed=wm.encoder(data)
        states,_=wm.dynamics.observe(embed,data["action"],data["is_first"])
        out.append(wm.dynamics.get_feat(states)[:,-1])
    return torch.cat(out)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tag",required=True); ap.add_argument("--rssm_ckpt",required=True)
    ap.add_argument("--policy",required=True); ap.add_argument("--seed",type=int,default=0)
    a=ap.parse_args()
    d=np.load(HJ); V,xs,ys,ths=d["V"],d["grid_xs"],d["grid_ys"],d["grid_thetas"]
    rng=np.random.default_rng(a.seed)
    st=np.stack([rng.uniform(xs[0],xs[-1],N),rng.uniform(ys[0],ys[-1],N),rng.uniform(-np.pi,np.pi,N)],1).astype(np.float32)
    y=hj_interp(V,xs,ys,ths,st); safe=(y>=0).astype(int)
    cfg=DreamerConfig(); ec=Config().environment
    cfg.turnRate=ec.max_angular_velocity
    cfg.x_min,cfg.x_max=ec.world_bounds[0],ec.world_bounds[1]; cfg.y_min,cfg.y_max=ec.world_bounds[2],ec.world_bounds[3]
    cfg.size=[128,128]; cfg.filter_mode="cbf"; cfg.no_gp=False; cfg.wm_backend="dreamer"
    wm=build_dreamer_wm(cfg,a.rssm_ckpt,DEV)
    # feat size from one forward
    env=make_env(); f0=feats(wm,render_cond(env,st[:2],CONDS["red"]),st[:2,2])
    policy=load_ddpg(a.policy,int(f0.shape[1]),cfg,DEV)
    print(f"tag={a.tag} N={N} feat={f0.shape[1]} frac_safe={safe.mean():.3f}")
    for cname,cond in CONDS.items():
        f=feats(wm,render_cond(env,st,cond),st[:,2]); Vh=eval_V(policy,f)
        acc=float(((Vh>=0)==(safe==1)).mean()); auc=float(roc_auc_score(safe,Vh)) if 0<safe.mean()<1 else float("nan")
        print(f"KEY critic {a.tag} cond={cname} acc={acc:.3f} auc={auc:.3f}")
    print("done")


if __name__=="__main__":
    main()
