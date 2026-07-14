/* Tokenizer GLM-5.2 in C puro (byte-level BPE stile cl100k / tiktoken).
 * Replica fedele di tokenizer.json:
 *   - model.type = BPE, byte_fallback=false
 *   - pre_tokenizer: regex Split (pattern cl100k) + ByteLevel(add_prefix_space=false)
 *   - merges con rank = ordine nella lista; \p{L}/\p{N}/\s da tok_unicode.h
 *   - added_tokens (speciali e non) trattati come atomici in encode/decode
 * API:
 *   if (tok_load(&T, "tokenizer.json") != 0) ...;
 *   int n = tok_encode(&T, text, len, out_ids, max);
 *   int m = tok_decode(&T, ids, n, out_buf, max);
 *   tok_free(&T);
 */
#ifndef TOK_H
#define TOK_H
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include "json.h"
#include "tok_unicode.h"

/* ---------- hash map (chiavi binarie con lunghezza) ---------- */
typedef struct { const char *k; int klen; int v; int used; } ment;
typedef struct { ment *e; int cap; } hmap;
static uint64_t tk_fnv(const char *s, int n){ uint64_t h=1469598103934665603ULL;
    for(int i=0;i<n;i++){ h^=(unsigned char)s[i]; h*=1099511628211ULL; } return h; }
static void hm_init(hmap *m, int cap){ m->cap=cap; m->e=(ment*)calloc(cap,sizeof(ment)); }
static void hm_put(hmap *m, const char *k, int klen, int v){
    uint64_t h=tk_fnv(k,klen)&(m->cap-1);
    while(m->e[h].used){ if(m->e[h].klen==klen && !memcmp(m->e[h].k,k,klen)){ m->e[h].v=v; return; } h=(h+1)&(m->cap-1); }
    m->e[h].k=k; m->e[h].klen=klen; m->e[h].v=v; m->e[h].used=1;
}
static int hm_get(hmap *m, const char *k, int klen){
    uint64_t h=tk_fnv(k,klen)&(m->cap-1);
    while(m->e[h].used){ if(m->e[h].klen==klen && !memcmp(m->e[h].k,k,klen)) return m->e[h].v; h=(h+1)&(m->cap-1); }
    return -1;
}

typedef struct { char *str; int len; int id; } Special;
typedef struct {
    hmap vocab;          /* stringa byte-level -> id */
    hmap merges;         /* "left\0right" -> rank */
    char **id2str; int *id_added; int n_ids;   /* id -> stringa; id_added=1 se added-token (output letterale) */
    Special *sp; int nsp;                       /* added tokens, ordinati per lunghezza decrescente */
    uint32_t byte2cp[256]; int byte2cp_len[256]; char byte2str[256][3];
    int16_t cp2byte[1024];
} Tok;

/* ---------- UTF-8 ---------- */
static int u8_next(const unsigned char *s, int len, int i, uint32_t *cp){
    unsigned char c=s[i];
    if(c<0x80){ *cp=c; return 1; }
    if((c>>5)==0x6 && i+1<len){ *cp=((c&0x1F)<<6)|(s[i+1]&0x3F); return 2; }
    if((c>>4)==0xE && i+2<len){ *cp=((c&0x0F)<<12)|((s[i+1]&0x3F)<<6)|(s[i+2]&0x3F); return 3; }
    if((c>>3)==0x1E && i+3<len){ *cp=((c&0x07)<<18)|((s[i+1]&0x3F)<<12)|((s[i+2]&0x3F)<<6)|(s[i+3]&0x3F); return 4; }
    *cp=c; return 1;   /* byte invalido: trattato come singolo */
}
static int u8_put(char *o, uint32_t cp){
    if(cp<0x80){ o[0]=cp; return 1; }
    if(cp<0x800){ o[0]=0xC0|(cp>>6); o[1]=0x80|(cp&0x3F); return 2; }
    if(cp<0x10000){ o[0]=0xE0|(cp>>12); o[1]=0x80|((cp>>6)&0x3F); o[2]=0x80|(cp&0x3F); return 3; }
    o[0]=0xF0|(cp>>18); o[1]=0x80|((cp>>12)&0x3F); o[2]=0x80|((cp>>6)&0x3F); o[3]=0x80|(cp&0x3F); return 4;
}

/* ---------- mappa byte<->unicode di GPT-2/ByteLevel ---------- */
static void tk_build_bytemap(Tok *T){
    for(int i=0;i<1024;i++) T->cp2byte[i]=-1;
    int isdir[256]; memset(isdir,0,sizeof(isdir));
    for(int b=33;b<=126;b++) isdir[b]=1;
    for(int b=161;b<=172;b++) isdir[b]=1;
    for(int b=174;b<=255;b++) isdir[b]=1;
    int n=0;
    for(int b=0;b<256;b++){
        uint32_t cp = isdir[b] ? (uint32_t)b : (uint32_t)(256+n);
        if(!isdir[b]) n++;
        T->byte2cp[b]=cp;
        T->byte2cp_len[b]=u8_put(T->byte2str[b], cp);
        if(cp<1024) T->cp2byte[cp]=b;
    }
}

/* ---------- caricamento tokenizer.json ---------- */
static char *tk_read_file(const char *path, long *out_n){
    FILE *f=fopen(path,"rb"); if(!f) return NULL;
    if(fseek(f,0,SEEK_END)!=0){ fclose(f); return NULL; }
    long n=ftell(f); if(n<0){ fclose(f); return NULL; }
    if(fseek(f,0,SEEK_SET)!=0){ fclose(f); return NULL; }
    char *b=malloc((size_t)n+1); if(!b){ fclose(f); return NULL; }
    if(fread(b,1,(size_t)n,f)!=(size_t)n){ free(b); fclose(f); return NULL; }
    b[n]=0; fclose(f); if(out_n)*out_n=n; return b;
}
static int cmp_sp_len(const void *a, const void *b){ return ((const Special*)b)->len - ((const Special*)a)->len; }

static void hm_free(hmap *m, int free_keys){
    if(!m || !m->e) return;
    if(free_keys){
        for(int i=0;i<m->cap;i++)
            if(m->e[i].used) free((void *)m->e[i].k);
    }
    free(m->e);
    m->e=NULL; m->cap=0;
}

static void tok_free(Tok *T){
    if(!T) return;
    hm_free(&T->vocab, 1);
    hm_free(&T->merges, 1);
    if(T->sp){
        for(int i=0;i<T->nsp;i++) free(T->sp[i].str);
        free(T->sp);
    }
    free(T->id2str);
    free(T->id_added);
    memset(T,0,sizeof(*T));
}

/* Returns 0 on success, -1 on failure (Tok left zeroed). Does not exit. */
static int tok_load(Tok *T, const char *path){
    if(!T || !path) return -1;
    memset(T,0,sizeof(*T));
    tk_build_bytemap(T);
    long fn=0; char *buf=tk_read_file(path,&fn);
    if(!buf){ fprintf(stderr,"tokenizer: cannot read %s\n", path); return -1; }
    char *arena=NULL; jval *root=json_parse(buf,&arena);
    free(buf);
    free(arena);
    if(!root){ fprintf(stderr,"tokenizer.json: parse failed\n"); return -1; }
    jval *model=json_get(root,"model");
    jval *vocab=json_get(model,"vocab");
    jval *merges=json_get(model,"merges");
    jval *added=json_get(root,"added_tokens");
    if(!vocab||!merges){
        fprintf(stderr,"tokenizer.json: missing model.vocab/merges\n");
        json_free(root);
        return -1;
    }

    /* id massimo per dimensionare id2str */
    int maxid=0;
    for(int i=0;i<vocab->len;i++){ int id=(int)vocab->kids[i]->num; if(id>maxid)maxid=id; }
    if(added) for(int i=0;i<added->len;i++){
        jval *idv=json_get(added->kids[i],"id");
        if(idv){ int id=(int)idv->num; if(id>maxid)maxid=id; }
    }
    T->n_ids=maxid+1;
    T->id2str=calloc((size_t)T->n_ids,sizeof(char*));
    T->id_added=calloc((size_t)T->n_ids,sizeof(int));
    if(!T->id2str || !T->id_added){ tok_free(T); json_free(root); return -1; }

    /* vocab: stringa -> id  (capacita' potenza di 2, ~2-3x); own copies. */
    int vc=1; while(vc < vocab->len*2) vc<<=1;
    hm_init(&T->vocab, vc);
    if(!T->vocab.e){ tok_free(T); json_free(root); return -1; }
    for(int i=0;i<vocab->len;i++){
        const char *k=vocab->keys[i]; int id=(int)vocab->kids[i]->num;
        size_t klen=strlen(k);
        char *owned=malloc(klen+1);
        if(!owned){ tok_free(T); json_free(root); return -1; }
        memcpy(owned,k,klen+1);
        hm_put(&T->vocab, owned, (int)klen, id);
        if(id>=0 && id<T->n_ids) T->id2str[id]=owned;
    }
    /* merges: HF accepts either [left,right] pairs or "left right" strings.
     * Internally both become "left\0right" -> rank=i. */
    int mc=1; while(mc < merges->len*2) mc<<=1;
    hm_init(&T->merges, mc);
    if(!T->merges.e){ tok_free(T); json_free(root); return -1; }
    for(int i=0;i<merges->len;i++){
        jval *pr=merges->kids[i];
        const char *l=NULL, *r=NULL; int ll=0, rl=0;
        if(pr->t==J_ARR && pr->len==2 && pr->kids[0]->t==J_STR && pr->kids[1]->t==J_STR){
            l=pr->kids[0]->str; r=pr->kids[1]->str;
            ll=(int)strlen(l); rl=(int)strlen(r);
        }else if(pr->t==J_STR){
            const char *separator=strchr(pr->str,' ');
            if(!separator) continue;
            l=pr->str; ll=(int)(separator-l); r=separator+1; rl=(int)strlen(r);
        }else continue;
        char *key=malloc((size_t)ll+1+(size_t)rl);
        if(!key){ tok_free(T); json_free(root); return -1; }
        memcpy(key,l,(size_t)ll); key[ll]=0; memcpy(key+ll+1,r,(size_t)rl);
        hm_put(&T->merges, key, ll+1+rl, i);
    }
    /* added tokens (speciali e non): atomici, output letterale */
    if(added){
        T->nsp=added->len; T->sp=calloc((size_t)T->nsp,sizeof(Special));
        if(!T->sp){ tok_free(T); json_free(root); return -1; }
        for(int i=0;i<added->len;i++){
            jval *a=added->kids[i];
            jval *content_v=json_get(a,"content");
            jval *id_v=json_get(a,"id");
            if(!content_v || !id_v || content_v->t!=J_STR){
                tok_free(T); json_free(root); return -1;
            }
            char *content=content_v->str; int id=(int)id_v->num;
            size_t clen=strlen(content);
            char *owned=malloc(clen+1);
            if(!owned){ tok_free(T); json_free(root); return -1; }
            memcpy(owned,content,clen+1);
            T->sp[i].str=owned; T->sp[i].len=(int)clen; T->sp[i].id=id;
            if(id>=0 && id<T->n_ids){ T->id2str[id]=owned; T->id_added[id]=1; }
        }
        qsort(T->sp,T->nsp,sizeof(Special),cmp_sp_len);   /* match piu' lungo per primo */
    }
    json_free(root);
    return 0;
}

/* ---------- BPE su un pezzo: byte grezzi [a,b) -> id appesi a out ---------- */
static void bpe_piece(Tok *T, const unsigned char *p, int a, int b, int *out, int *no, int max){
    int nb=b-a;
    /* stringa byte-level (concatenazione di byte2str): <=2 byte per byte di input */
    char *s=malloc(2*nb+1); int sl=0;
    for(int i=a;i<b;i++){ int bb=p[i]; memcpy(s+sl,T->byte2str[bb],T->byte2cp_len[bb]); sl+=T->byte2cp_len[bb]; }
    s[sl]=0;
    /* ignore_merges: se l'intero pezzo e' un token, emettilo diretto */
    int whole=hm_get(&T->vocab,s,sl);
    if(whole>=0){ if(*no<max) out[(*no)++]=whole; free(s); return; }
    /* simboli iniziali = codepoint della stringa byte-level */
    int *soff=malloc((sl+1)*sizeof(int)), *slen=malloc((sl+1)*sizeof(int)); int ns=0;
    for(int i=0;i<sl;){ uint32_t cp; int k=u8_next((const unsigned char*)s,sl,i,&cp);
        soff[ns]=i; slen[ns]=k; ns++; i+=k; }
    char *kbuf=malloc(2*sl+2);
    for(;;){
        int best=INT_MAX, bp=-1;
        for(int i=0;i+1<ns;i++){
            int ll=slen[i], rl=slen[i+1];
            memcpy(kbuf,s+soff[i],ll); kbuf[ll]=0; memcpy(kbuf+ll+1,s+soff[i+1],rl);
            int rk=hm_get(&T->merges,kbuf,ll+1+rl);
            if(rk>=0 && rk<best){ best=rk; bp=i; }
        }
        if(bp<0) break;
        slen[bp]=soff[bp+1]+slen[bp+1]-soff[bp];          /* fonde bp e bp+1 (contigui in s) */
        for(int j=bp+1;j<ns-1;j++){ soff[j]=soff[j+1]; slen[j]=slen[j+1]; }
        ns--;
    }
    for(int i=0;i<ns;i++){
        int id=hm_get(&T->vocab,s+soff[i],slen[i]);
        if(id>=0 && *no<max) out[(*no)++]=id;
    }
    free(s); free(soff); free(slen); free(kbuf);
}

/* ---------- pre-tokenizer regex (pattern cl100k) su una porzione di testo ----------
 * Decodifica i codepoint, applica le alternative IN ORDINE, e per ogni pezzo chiama bpe_piece. */
static void pretok_chunk(Tok *T, const unsigned char *p, int a, int b, int *out, int *no, int max){
    int nb=b-a; if(nb<=0) return;
    uint32_t *cp=malloc((nb+1)*sizeof(uint32_t)); int *off=malloc((nb+2)*sizeof(int)); int n=0;
    for(int i=a;i<b;){ uint32_t c; int k=u8_next(p,b,i,&c); off[n]=i; cp[n]=c; n++; i+=k; }
    off[n]=b;
    #define ISNL(c) ((c)=='\r'||(c)=='\n')
    #define LOW(c) (((c)>='A'&&(c)<='Z')?((c)+32):(c))
    int i=0;
    while(i<n){
        int start=i; uint32_t c=cp[i];
        /* 1) (?i:'s|'t|'re|'ve|'m|'ll|'d) */
        if(c=='\'' && i+1<n){
            uint32_t d=LOW(cp[i+1]);
            if(i+2<n){ uint32_t d2=LOW(cp[i+2]);
                if((d=='r'&&d2=='e')||(d=='v'&&d2=='e')||(d=='l'&&d2=='l')){ i+=3; bpe_piece(T,p,off[start],off[i],out,no,max); continue; } }
            if(d=='s'||d=='t'||d=='m'||d=='d'){ i+=2; bpe_piece(T,p,off[start],off[i],out,no,max); continue; }
        }
        /* 2) [^\r\n\p{L}\p{N}]? \p{L}+ */
        {
            int j=i;
            if(!is_L(c) && !ISNL(c) && !is_N(c)){ if(j+1<n && is_L(cp[j+1])) j++; else j=-1; }
            if(j>=0){
                if(is_L(cp[j])){ while(j<n && is_L(cp[j])) j++; i=j; bpe_piece(T,p,off[start],off[i],out,no,max); continue; }
            }
        }
        /* 3) \p{N}{1,3} */
        if(is_N(c)){ int j=i,k=0; while(j<n && is_N(cp[j]) && k<3){ j++; k++; } i=j; bpe_piece(T,p,off[start],off[i],out,no,max); continue; }
        /* 4) ' ?[^\s\p{L}\p{N}]+[\r\n]*' */
        {
            int j=i;
            if(c==' ' && j+1<n && !is_S(cp[j+1]) && !is_L(cp[j+1]) && !is_N(cp[j+1])) j++;
            if(j<n && !is_S(cp[j]) && !is_L(cp[j]) && !is_N(cp[j])){
                while(j<n && !is_S(cp[j]) && !is_L(cp[j]) && !is_N(cp[j])) j++;
                while(j<n && ISNL(cp[j])) j++;
                i=j; bpe_piece(T,p,off[start],off[i],out,no,max); continue;
            }
        }
        /* 5) \s*[\r\n]+  -> run di whitespace fino all'ultimo newline contiguo */
        {
            int r=i; while(r<n && is_S(cp[r])) r++;
            if(r>i){ int last=-1; for(int j=i;j<r;j++) if(ISNL(cp[j])) last=j;
                if(last>=0){ i=last+1; bpe_piece(T,p,off[start],off[i],out,no,max); continue; }
                /* 6) \s+(?!\S): se seguito da non-spazio lascia l'ultimo ws, altrimenti prendi tutto */
                int end = (r<n) ? r-1 : r;
                if(end<=i) end=i+1;                 /* \s+ minimo 1 (fallback alt 7) */
                i=end; bpe_piece(T,p,off[start],off[i],out,no,max); continue;
            }
        }
        i++;  /* salvagente: non dovrebbe accadere */
        bpe_piece(T,p,off[start],off[i],out,no,max);
    }
    #undef ISNL
    #undef LOW
    free(cp); free(off);
}

/* ---------- encode: testo -> id (split sugli added token, poi pretok+BPE) ---------- */
static int tok_encode(Tok *T, const char *text, int len, int *out, int max){
    const unsigned char *p=(const unsigned char*)text; int no=0; int i=0;
    while(i<len){
        /* prossima occorrenza di un added-token a partire da >= i (match piu' lungo) */
        int hitpos=-1, hitlen=0, hitid=-1;
        for(int j=i;j<len && hitpos<0;j++){
            for(int k=0;k<T->nsp;k++){
                int sl=T->sp[k].len;
                if(sl>0 && j+sl<=len && !memcmp(p+j,T->sp[k].str,sl)){ hitpos=j; hitlen=sl; hitid=T->sp[k].id; break; }
            }
        }
        int chunk_end = (hitpos<0) ? len : hitpos;
        if(chunk_end>i) pretok_chunk(T,p,i,chunk_end,out,&no,max);
        if(hitpos<0) break;
        if(no<max) out[no++]=hitid;
        i=hitpos+hitlen;
    }
    return no;
}

/* id di un added-token dato il suo contenuto (es. "<|endoftext|>"); -1 se assente */
static int tok_id_of(Tok *T, const char *content){
    for(int i=0;i<T->nsp;i++) if(!strcmp(T->sp[i].str,content)) return T->sp[i].id;
    return -1;
}

/* ---------- decode: id -> testo (byte-level inverso; added token letterali) ---------- */
static int tok_decode(Tok *T, const int *ids, int n, char *out, int max){
    int o=0;
    for(int i=0;i<n;i++){
        int id=ids[i]; if(id<0||id>=T->n_ids||!T->id2str[id]) continue;
        const char *s=T->id2str[id];
        if(T->id_added[id]){ int l=(int)strlen(s); for(int j=0;j<l && o<max;j++) out[o++]=s[j]; continue; }
        int sl=(int)strlen(s);
        for(int j=0;j<sl;){ uint32_t c; int k=u8_next((const unsigned char*)s,sl,j,&c); j+=k;
            if(c<1024 && T->cp2byte[c]>=0 && o<max) out[o++]=(char)(unsigned char)T->cp2byte[c]; }
    }
    if(o<max) out[o]=0;
    return o;
}

#endif
