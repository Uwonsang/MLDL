import os
from shutil import copyfile

from pytube import YouTube

source_path = "C:/Users/CILAB_THP/Desktop/Verified_Normed/"

# 파일 저장 위치
dst_path = "C:/Users/wonsang/Desktop/music_samples/"

file_names = {
    "amusing": ["DA-m6OrZZyk_0.mp3","Bk2oy5NQY5I_35.mp3","mnipB_8Br8U_16.mp3","7s21gNbCYXk_1.mp3","Ag1o3koTLWM_5.mp3","KcSzSpWdpwE_1.mp3","T5DnqW3F57E_4.mp3","d-diB65scQU_1.mp3","woPff-Tpkns_0.mp3"],
    "angry": ["1qKS51qh4OY_65.mp3","OYjZK_6i37M_44.mp3","f5Yj_JPtrI8_198.mp3","sMzAotfj6Cg_45.mp3","woAcXSMyCEw_30.mp3","D_-a3SIjNKHHg_260.mp3","If9VmBlp82c_45.mp3","L78yVFeyvRo_32.mp3","Ph-CA_tu5KA_8.mp3"],
    "annoying": ["ZHUGeuMm45w_24.mp3","YI67GAQYsRc_465.mp3","0dtkfpTwDxU_1.mp3","D_-a3SIjNKHHg_260.mp3","O8j1bI0zzeQ_4.mp3","e5uz85tLoJo_2.mp3","pG1dtbsYn2M_225.mp3","D_-gPuH1yeZ08_314.mp3","Y_IS_ccjkZE_110.mp3"],
    "anxious&tense": ["Epic Suspense Action Music - Tracked-qHkTjuRWY18&1.mp3","Mordor_FootstepsDoom_short.mp3","VYz3VIQeET8_47.mp3","IAIan0h85Xg_4.mp3","BhMkJ9pvsFI_30.mp3","XjjbsbcQZy8_20.mp3","2AZ1hJzKocI_5.mp3","cWjOaZFXo3s_25.mp3","D7BoKuu0MNk_1.mp3"],
    "awe-inspiring&amazing": ["dnA__dfGxCA_1334.mp3","499.m4a.mp3","RX2y3ncQGR4_132.mp3","7BhGWO01pIs_38.mp3","GRxofEmo3HA_15.mp3","Fellowship_WhiteRider_short.mp3","Four Seasons ~ Vivaldi-GRxofEmo3HA&2.mp3","j8KL63r9Zcw_105.mp3","497.m4a.mp3"],
    "beautiful": ["2423.m4a.mp3","ehmYO0qoCm4_100.mp3","NliYy7iqh-U_29.mp3","NlprozGcs80_127.mp3","k_UOuSklNL4_2.mp3","521.m4a.mp3","0Uv4lJsu0Jw_565.mp3","2NuuhRrRQZo_3.mp3","691.m4a.mp3"],
    "bittersweet": ["UWZlqjzZRDo_170.mp3","M83 - Wait_Rw7aMVvPDmc_2.mp3","5anLPw0Efmo_5.mp3","XNSsv86lsok_2.mp3","mF3DCa4TbD0_151.mp3","xTs83Ej5nS8_226.mp3","cAVn71rNImI_10.mp3","IXigE1uceSI_24.mp3","aWIE0PX1uXk_70.mp3"],
    "calm&relaxing&serene": ["Pat Metheny_Charlie Haden - Love Theme (from 'Cinema Paradiso') HD1080-_s4zrg2z-Pq8_2.mp3","w-Rpa9TeKGc_455.mp3","9vdFHnl89Yk_3.mp3","aWIE0PX1uXk_22.mp3","2464.m4a.mp3","2341.m4a.mp3","Ri3WsPDi4MY_69.mp3","93CyZ1rt2vQ_1.mp3","QZbuj3RJcjI_3900.mp3"],
    "compassionate&sympathetic": ["pUZeSYsU0Uk_76.mp3","Kmzh43Kuyxk_0.mp3","2mREd6kvE_I_301.mp3","3nmxBiZQTvg_0.mp3","cV9pGpI1MDo_2.mp3","2513.m4a.mp3","8CeLmrQI2oU_65.mp3","aWIE0PX1uXk_115.mp3","ecd4MmrUKrw_20.mp3"],
    "dreamy": ["1FmqrHSAtoc_565.mp3","01 - Tree Dance&1.mp3","vnCf88XR-Yg_4306.mp3","EZVHjVbUP40_15.mp3","UPW8y6woTBI_10.mp3","c1EK4Emqw3g_294.mp3","w-Rpa9TeKGc_455.mp3","2118.m4a.mp3","Ly7uj0JwgKg_0.mp3"],
    "eerie&mysterious": ["HQoRXhS7vlU_3.mp3","8QpUGCXwOks_5.mp3","v2XR6zmZ4Ps_3083.mp3","4S-5ihzUkM4_121.mp3","D_-X1-uqDYa7w_1.mp3","1FH-q0I1fJY_20.mp3","hAAlDoAtV7Y_18.mp3","YYb9kSCkjE8_13.mp3","eAcx8UVc_Uk_8.mp3"],
    "energizing&pump-up": ["c7cmO25utTo_12.mp3","gzoEK545j64_46.mp3","CDl9ZMfj6aE_8.mp3","AEB6ibtdPZc_23.mp3","kdih7kjmQJw_53.mp3","D_-5XxJ4XBm50_101.mp3","362.m4a.mp3","bU0fmaAYyxU_203.mp3","oLahU5721ks_38.mp3"],
    "entrancing": ["ootz7HsxrK8_2.mp3","2vMH8lITTCE_10.mp3","2835.m4a.mp3","Porches - Underwater_XxfNqvoXRug_3.mp3","XvsbwJv3_YY_38.mp3","ME_ascd_FallOfMen_short.mp3","bU0fmaAYyxU_203.mp3","uhdBQ-pUyO4_5.mp3","_9WUB1QF_4U_59.mp3"],
    "erotic&desirous": ["gNDYBPCJPqU_17.mp3","Snails and Barry White_VMEJMjGgeus_1.mp3","Bootsy Collins - I'd Rather Be With You-0tgYr03o3dE_1.mp3","w6ND5PZxA&2.mp3","Marvin Gaye - Lets get it on_x6QZn9xiuOE_1.mp3","D'angelo - Untitled (How Does It Feel)_-7hbqxnhzU8_2.mp3","k3Fa4lOQfbA_151.mp3","Bootsy Collins  -  Munchies For Your Love-kVYwZ1hgNiU&1.mp3","w6ND5PZxA&1.mp3"],
    "euphoric&ecstatic": ["ootz7HsxrK8_15.mp3","32LB2DR_JM0_86.mp3","cg_kcK6o-5U_59.mp3","fVtUFOSyIyw_2.mp3","ynNNtdni2h8_5.mp3","y6120QOlsfU_16.mp3","Run The Heart - Sleigh Bells_4J2MIv7B4bI_2.mp3","9aZFcosBTaQ_182.mp3","kdih7kjmQJw_53.mp3"],
    "exciting": ["ynNNtdni2h8_5.mp3","reTx5sqvVJ4_155.mp3","c7cmO25utTo_12.mp3","gzeOWnnSNjg_4.mp3","tLTGs4fqxBk_10.mp3","Four Seasons ~ Vivaldi-GRxofEmo3HA&2.mp3","QYqsevBV97g_40.mp3","kdih7kjmQJw_53.mp3","V3n78uCYdso_0.mp3"],
    "goose bumps": ["2-Hour Anime Mix - Best Of Anime Soundtracks _ Emotional Ride - Epic Music-S7OCzDNeENg&5.mp3","Four Seasons ~ Vivaldi-GRxofEmo3HA&3.mp3","ae8FyeVc7qk_0.mp3","BX3bN5YeiQs_1.mp3","Mordor_Ringwraiths_short.mp3","Eegs84CdzTc_450.mp3","Epic Suspense Action Music - Tracked-qHkTjuRWY18&1.mp3","hAAlDoAtV7Y_18.mp3","6TUeUL7EW9M_1.mp3"],
    "indignant&defiant": ["9gAfe7nQWCg_1.mp3","ykhG9g6UwB0_9.mp3","59EBD48gNZs_15.mp3","9PbeYAvGZA4_38.mp3","9P4GsF1zdzM_40.mp3","1qKS51qh4OY_65.mp3","bFQNSGR2_TA_3.mp3","CDl9ZMfj6aE_8.mp3","Rage Against The Machine - Killing In the Name_bWXazVhlyxQ_1.mp3"],
    "joyful&cheerful": ["6wlbB1PTzJU_0.mp3","jNvdh9Q6Qs0_0.mp3","d-diB65scQU_1.mp3","F3TzxXfoEHo_0.mp3","ohB3KxbkuNk_251.mp3","ZBR2G-iI3-I_79.mp3","jNvdh9Q6Qs0_13.mp3","vFe5N7bvXXI_10.mp3","us67L7V6VIg_1033.mp3"],
    "nauseating&revolting": ["ckZlj2p8W9M_11.mp3","61Mgnw-4qm4_14.mp3","3419.m4a.mp3","O8j1bI0zzeQ_4.mp3","L_XJ_s5IsQc_448.mp3","144.m4a.mp3","2NV_zHmgw3U_31.mp3","e5uz85tLoJo_2.mp3","E8-WHslFhbU_89.mp3"],
    "painful": ["D_-a3SIjNKHHg_260.mp3","9f0l4djubOM_0.mp3","8UgxSc4sgtI_5.mp3","efioLa0JDcY_371.mp3","ckZlj2p8W9M_11.mp3","2NV_zHmgw3U_31.mp3","04F4xlWSFh0_75.mp3","ZHUGeuMm45w_24.mp3","wkyGQnZNLYY_4.mp3"],
    "proud&strong": ["e9vrfEoc8_g_10.mp3","27mB8verLK8_58.mp3","I33u_EHLI3w_8.mp3","sMzAotfj6Cg_45.mp3","cYVL3LkPBXA_759.mp3","9jK-NcRmVcw_234.mp3","jv2WJMVPQi8_4.mp3","1qKS51qh4OY_65.mp3","woAcXSMyCEw_30.mp3"],
    "romantic&loving": ["bYOE4XnrNeo_80.mp3","1BdPDaFXcEo_1.mp3","EFZeC-1wYmI_82.mp3","D'angelo - Untitled (How Does It Feel)_-7hbqxnhzU8_1.mp3","A3yCcXgbKrE_0.mp3","Marvin Gaye - Lets get it on_x6QZn9xiuOE_1.mp3","U6Xs-dh83i4_9.mp3","NiIiRGP4g7s_6658.mp3","qiiyq2xrSI0_200.mp3"],
    "sad&depressing": ["YQHsXMglC9A_75.mp3","XNSsv86lsok_2.mp3","pUZeSYsU0Uk_85.mp3","Wagner Leitmotives - 10 - Renunciation of Love-zerjsBZgDsE_short.mp3","FOVzbp5yVsU_20.mp3","2478.m4a.mp3","VIkr-RcQ4l0_754.mp3","2NuuhRrRQZo_3.mp3","aWIE0PX1uXk_46.mp3"],
    "scary&fearful": ["R3WwcsjWPIQ_23.mp3","6vtsKGzGVK4_4.mp3","amZQdMtjNA8_12.mp3","4S-5ihzUkM4_2005.mp3","6vtsKGzGVK4_20.mp3","BX3bN5YeiQs_1.mp3","6TUeUL7EW9M_1.mp3","Wagner Leitmotives - 11 - Love Tragedy-Wl-aAKO_QYA_short.mp3","Isengard_Grima_short.mp3"],
    "tender&longing": ["NiIiRGP4g7s_6658.mp3","bYOE4XnrNeo_80.mp3","447yaU_4DF8_15.mp3","HyrWd_gfQNQ_39.mp3","bcTRzNTwc1o_0.mp3","Vt2YIpZWBqA_6.mp3","Leslie Feist - Lover's Spit (Broken Social Scene)_P4vSxFIc0xo_1.mp3","lcOxhH8N3Bo_0.mp3","UWZlqjzZRDo_200.mp3"],
    "transcendent&mystical": ["Gregorian Chants from Assisi -  Medieval Lauds-eC6OKIYXBxQ&1.mp3","4F-CpE73o2M_1.mp3","pFDV04HObg0_10_fix.mp3","YeaGUfZM5hs_179.mp3","3143.m4a.mp3","JxUP9p2x1Cw_90.mp3","Gregorian Chants from Assisi -  Medieval Lauds-eC6OKIYXBxQ&2.mp3","jTW4DdVYKZs_5.mp3","GPXWflXWG8k_125.mp3"],
    "triumphant&heroic": ["DbXTYVrJiOQ_1.mp3","Wagner Leitmotives - 33 - Thunder-rqPNg_smDE4.mp3","2492.m4a.mp3","nOr0na6mKJQ_70.mp3","Fellowship_Aragorn_short.mp3","3YOYlgvI1uE_23.mp3","I33u_EHLI3w_8.mp3","OJk_1C7oRZg_4020.mp3","27mB8verLK8_58.mp3"]
}
dst_folder_names = ["amusing","angry","annoying","anxious&tense","awe-inspiring&amazing","beautiful","bittersweet","calm&relaxing&serene","compassionate&sympathetic","dreamy","eerie&mysterious","energizing&pump-up","entrancing","erotic&desirous","euphoric&ecstatic","exciting","goose bumps","indignant&defiant","joyful&cheerful","nauseating&revolting","painful","proud&strong","romantic&loving","sad&depressing","scary&fearful","tender&longing","transcendent&mystical","triumphant&heroic"]

def find(name, path):
    for root, dirs, files in os.walk(path):
        for filename in files:    
            if name in filename:
                return os.path.join(root, filename)

# for dst_folder_name in dst_folder_names:
#     for file_name in file_names[dst_folder_name]:
#         src = source_path + file_name
#         dst = dst_path + dst_folder_name + "/" + file_name
#         copyfile(src, dst)
cnt = 1
for dst_folder_name in dst_folder_names:
    print("DownLoading..."+str(cnt)+"/"+str(len(dst_folder_names)))
    cnt += 1
    for file_name in file_names[dst_folder_name]:
        file_name = file_name[:-4]
        url = "https://youtube.com/watch?v="+file_name
        dst = dst_path + dst_folder_name + "/" + file_name

        try:
            YouTube(url).streams.filter(only_audio=True).last().download(dst)
            # YouTube(url).streams.first().filter(only_audio=True).fisrt().download(dst)
        except:
            print("there is no url :"+file_name)