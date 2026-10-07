"""OOD eval of a trained reachability critic (the VALUE FUNCTION): sign-acc / AUC
of V_latent (critic) vs HJ V* under obstacle-appearance shift. JEPA/LEWM.

Ground truth = HJ V* (reachability), NOT the instantaneous margin. For one trained
critic (policy.pth) + its WM, score V's sign vs (V*>=0) on each appearance
condition (red/purple/diamond/rot90) — red = in-dist, others = zero-shot.
Usage (from src/latent_cbf): critic_ood_eval.py --tag <t> --lewm_ckpt <w> --margin_ckpt <m> --policy <p.pth>
"""
import sys, argparse
from pathlib import Path
import numpy as np, torch
_THIS = Path(__file__).resolve()
for p in [str(_THIS.parents[3]), str(_THIS.parents[2]), str(_THIS.parents[1]), str(_THIS.parent)]:
    if p not in sys.path: sys.path.append(p)
for _p in ("/home/seongbin/latent", "/home/seongbin/latent/le-wm"):
    if _p not in sys.path: sys.path.append(_p)
import gym as old_gym  # noqa: F401
import gymnasium as gym  # noqa: F401
from PIL import Image, ImageDraw
from sklearn.metrics import roc_auc_score
from configs import DreamerConfig, Config
from latent_cbf.dubins.dubins_env import DubinsEnv
from latent_cbf.adapters import LEWMWorldModel
from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import Actor, Critic
from PyHJ.policy import avoid_DDPGPolicy_annealing as DDPGPolicy
from PyHJ.data import Batch

DEV, HJ, N = "cuda:0", "/home/seongbin/latent/latent_cbf/results/hj_truth.npz", 8000
CONDS = {"red":    dict(obstacle="red",    shape="circle",  rot=0),
         "purple": dict(obstacle="purple", shape="circle",  rot=0),
         "diamond":dict(obstacle="red",    shape="diamond", rot=0),
         "rot90":  dict(obstacle="red",    shape="circle",  rot=90)}


def hj_interp(V, xs, ys, ths, pts):
    nx, ny, nth = V.shape
    fx = np.clip((pts[:,0]-xs[0])/(xs[1]-xs[0]), 0, nx-1-1e-6)
    fy = np.clip((pts[:,1]-ys[0])/(ys[1]-ys[0]), 0, ny-1-1e-6)
    th = np.arctan2(np.sin(pts[:,2]), np.cos(pts[:,2])); ft = (th-ths[0])/(ths[1]-ths[0])
    i0,j0 = fx.astype(int), fy.astype(int); k0 = np.floor(ft).astype(int) % nth
    i1,j1 = np.minimum(i0+1,nx-1), np.minimum(j0+1,ny-1); k1=(k0+1)%nth
    wx,wy,wt = fx-i0, fy-j0, np.clip(ft-np.floor(ft),0,1); out=0.0
    for di,wi in ((i0,1-wx),(i1,wx)):
        for dj,wj in ((j0,1-wy),(j1,wy)):
            for dk,wk in ((k0,1-wt),(k1,wt)):
                out = out + V[di,dj,dk]*wi*wj*wk
    return out.astype(np.float32)


def load_ddpg(policy_path, feat_size, cfg, device):
    ss, as_ = (feat_size,), (1,)
    cnet = Net(ss, as_, hidden_sizes=cfg.critic_net, norm_layer=torch.nn.LayerNorm,
               activation=torch.nn.ReLU, concat=True, device=device)
    critic = Critic(cnet, device=device).to(device)
    copt = torch.optim.AdamW(critic.parameters(), lr=cfg.critic_lr)
    anet = Net(ss, hidden_sizes=cfg.control_net, activation=torch.nn.ReLU, device=device)
    actor = Actor(anet, as_, max_action=1.0, device=device).to(device)
    aopt = torch.optim.AdamW(actor.parameters(), lr=cfg.actor_lr)
    from PyHJ.exploration import GaussianNoise
    from gymnasium import spaces as gsp
    policy = DDPGPolicy(critic, copt, tau=cfg.tau, gamma=cfg.gamma_pyhj,
                        exploration_noise=GaussianNoise(sigma=cfg.exploration_noise),
                        reward_normalization=cfg.rew_norm, estimation_step=cfg.n_step,
                        action_space=gsp.Box(low=-1, high=1, shape=as_, dtype=np.float32),
                        actor=actor, actor_optim=aopt, actor_gradient_steps=cfg.actor_gradient_steps)
    policy.load_state_dict(torch.load(policy_path, map_location=device)); policy.eval()
    return policy


@torch.no_grad()
def eval_V(policy, feat):
    b = Batch(obs=feat, info=Batch())
    act = policy(b, model="actor_old").act
    return policy.critic_old(feat, act).view(-1).detach().cpu().numpy()


def render_diamond(env):
    scale=4; h=(env.image_size[0]*scale, env.image_size[1]*scale)
    img=Image.new("RGB",h,env.colors["background"]); d=ImageDraw.Draw(img)
    def w2p(c):
        x,y=c; return (int((x-env.x_min)/(env.x_max-env.x_min)*h[0]), int((env.y_max-y)/(env.y_max-env.y_min)*h[1]))
    for ox,oy,r in env.obstacles:
        cx,cy=w2p((ox,oy)); rp=r/(env.x_max-env.x_min)*h[0]
        d.polygon([(cx,cy-rp),(cx+rp,cy),(cx,cy+rp),(cx-rp,cy)],fill=env.colors["obstacle"])
    gc=w2p(env.goal_position); gr=env.goal_radius/(env.x_max-env.x_min)*h[0]
    d.ellipse([(gc[0]-gr,gc[1]-gr),(gc[0]+gr,gc[1]+gr)],fill=env.colors["goal"])
    env._draw_agent(d,w2p(env.state[:2]),float(env.state[2]),scale)
    return np.array(img.resize(env.image_size, Image.Resampling.LANCZOS))


def render_cond(env, st, cond):
    env.colors={"background":"white","agent":"blue","goal":"green","obstacle":cond["obstacle"]}
    dia=cond["shape"]=="diamond"; rot=cond.get("rot",0); imgs=[]
    for s in st:
        env.state=s.astype(np.float32)
        im=render_diamond(env) if dia else env.render()
        if rot: im=np.ascontiguousarray(np.rot90(im,k=rot//90))
        imgs.append(im)
    return np.stack(imgs,0)


@torch.no_grad()
def feats(wm, imgs, bs=128):
    out=[]
    for i in range(0,imgs.shape[0],bs):
        ch=imgs[i:i+bs][:,None]
        batch={"image":ch,"action":np.zeros((ch.shape[0],1,1),np.float32),
               "is_first":np.ones((ch.shape[0],1),np.float32),"is_terminal":np.zeros((ch.shape[0],1),np.float32)}
        data=wm.preprocess(batch); embed=wm.encoder(data); B=embed.shape[0]
        state={"deter":embed[:,-1],"stoch":torch.zeros(B,1,device=DEV)}
        out.append(wm.dynamics.get_feat(state))
    return torch.cat(out)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tag",required=True); ap.add_argument("--lewm_ckpt",required=True)
    ap.add_argument("--margin_ckpt",required=True); ap.add_argument("--policy",required=True)
    ap.add_argument("--seed",type=int,default=0)
    a=ap.parse_args()
    d=np.load(HJ); V,xs,ys,ths=d["V"],d["grid_xs"],d["grid_ys"],d["grid_thetas"]
    rng=np.random.default_rng(a.seed)
    st=np.stack([rng.uniform(xs[0],xs[-1],N),rng.uniform(ys[0],ys[-1],N),rng.uniform(-np.pi,np.pi,N)],1).astype(np.float32)
    y=hj_interp(V,xs,ys,ths,st); safe=(y>=0).astype(int)
    cfg=DreamerConfig(); cfg.lewm_ckpt_path=a.lewm_ckpt
    wm=LEWMWorldModel(cfg,a.lewm_ckpt).to(DEV)
    sd=torch.load(a.margin_ckpt,map_location=DEV)
    wm.heads["margin_gp"].load_state_dict(sd["margin_gp"]); wm.heads["margin_nogp"].load_state_dict(sd["margin_nogp"]); wm.eval()
    policy=load_ddpg(a.policy,int(wm.embed_dim),cfg,DEV)
    ec=Config().environment
    env=DubinsEnv(image_size=tuple(ec.image_size),world_bounds=tuple(ec.world_bounds),
                  max_angular_velocity=ec.max_angular_velocity,speed=ec.speed,dt=ec.dt,
                  obstacles=ec.get_obstacles_list(),goal_radius=ec.goal_radius,
                  collision_radius=ec.collision_radius,render_mode="rgb_array")
    print(f"tag={a.tag} N={N} frac_safe(V*>=0)={safe.mean():.3f}")
    for cname,cond in CONDS.items():
        f=feats(wm,render_cond(env,st,cond))
        Vh=eval_V(policy,f)
        acc=float(((Vh>=0)==(safe==1)).mean())
        auc=float(roc_auc_score(safe,Vh)) if 0<safe.mean()<1 else float("nan")
        print(f"KEY critic {a.tag} cond={cname} acc={acc:.3f} auc={auc:.3f}")
    print("done")


if __name__=="__main__":
    main()
