import { FileBlob, PresentationFile } from "@oai/artifact-tool";
const source = "/Users/wangyuhang/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.1/skills/artifact-template-simple-light-mode/assets/reference.pptx";
const p = await PresentationFile.importPptx(await FileBlob.load(source));
for (const i of [0,4,7,10,16,18,25]) {
  const s = p.slides.getItem(i);
  console.log(`===== source slide ${i+1} id=${s.id} =====`);
  for (let j=0;j<s.shapes.items.length;j++) {
    const sh=s.shapes.items[j];
    let txt="";
    try { txt=String(sh.text ?? "").replaceAll("\n"," | "); } catch {}
    console.log(j, JSON.stringify({id:sh.id,name:sh.name,placeholder:sh.placeholder?.type,position:sh.position,text:txt}));
  }
  for (let j=0;j<s.images.items.length;j++) {
    const im=s.images.items[j];
    console.log("image",j,JSON.stringify({id:im.id,name:im.name,frame:im.frame,fit:im.fit,crop:im.crop}));
  }
}
